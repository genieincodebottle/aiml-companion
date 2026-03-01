"""
main.py - IPL Match Predictor CLI
===================================

Production entry point for the IPL Match Predictor ML pipeline.
Supports running individual stages or the full pipeline end-to-end.

Usage
-----
    python main.py                    # Run full pipeline
    python main.py --stage clean      # Run only data cleaning
    python main.py --stage eda        # Run only EDA (saves figures)
    python main.py --stage features   # Run only feature engineering
    python main.py --stage hypothesis # Run only hypothesis tests
    python main.py --stage train      # Run only model training
    python main.py --stage evaluate   # Run only evaluation
    python main.py --help             # Show all options
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import clean_matches, clean_deliveries, clean_team_names, load_all
from src.eda import (
    plot_season_trends,
    plot_team_performance,
    plot_toss_analysis,
    plot_venue_stats,
    plot_win_margin_distribution,
    plot_player_of_match,
    save_figure,
)
from src.features import compute_elo_ratings, engineer_features
from src.hypothesis import run_toss_advantage_test, run_home_advantage_test
from src.models import (
    build_classifier,
    build_regressor,
    cross_validate_model,
    evaluate_model,
    get_feature_importances,
    prepare_features,
    split_data,
)
from src.evaluate import generate_report, print_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ipl-predictor")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def stage_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1: Load raw datasets."""
    logger.info("Loading datasets...")
    matches, deliveries = load_all()
    logger.info("Loaded %d matches, %d deliveries", len(matches), len(deliveries))
    return matches, deliveries


def stage_clean(
    matches: pd.DataFrame, deliveries: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2: Clean and standardize data."""
    logger.info("Cleaning data...")
    matches, deliveries = clean_team_names(matches, deliveries)
    matches = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)
    logger.info(
        "Cleaned: %d matches (nulls: %d), %d deliveries",
        len(matches),
        matches.isnull().sum().sum(),
        len(deliveries),
    )
    return matches, deliveries


def stage_eda(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    """Stage 3: Exploratory Data Analysis with saved figures."""
    logger.info("Running EDA...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    charts = {
        "season_trends": plot_season_trends(matches),
        "team_performance": plot_team_performance(matches),
        "toss_analysis": plot_toss_analysis(matches),
        "venue_stats": plot_venue_stats(matches),
        "win_margins": plot_win_margin_distribution(matches),
        "player_of_match": plot_player_of_match(matches),
    }

    for name, fig in charts.items():
        filepath = str(FIGURES_DIR / f"{name}.png")
        save_figure(fig, filepath)
        logger.info("Saved: %s", filepath)

    logger.info("EDA complete - %d charts saved", len(charts))


def stage_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Feature engineering with Elo ratings."""
    logger.info("Engineering features...")
    matches = compute_elo_ratings(matches)
    matches = engineer_features(matches)
    feature_cols = [c for c in matches.columns if c.startswith(("elo_", "momentum_", "h2h_", "home_"))]
    logger.info("Engineered %d new features: %s", len(feature_cols), feature_cols[:5])
    return matches


def stage_hypothesis(matches: pd.DataFrame) -> dict:
    """Stage 5: Statistical hypothesis testing."""
    logger.info("Running hypothesis tests...")
    results = {}

    toss_result = run_toss_advantage_test(matches)
    results["toss_advantage"] = toss_result
    logger.info(
        "Toss advantage test: p=%.4f, significant=%s",
        toss_result["p_value"],
        toss_result["significant"],
    )

    home_result = run_home_advantage_test(matches)
    results["home_advantage"] = home_result
    logger.info(
        "Home advantage test: p=%.4f, significant=%s",
        home_result["p_value"],
        home_result["significant"],
    )

    return results


def stage_train(
    matches: pd.DataFrame,
) -> tuple[object, object, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
    """Stage 6: Model training with cross-validation."""
    logger.info("Training models...")

    # Classification: predict match winner (team1 wins = 1)
    matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)
    valid = matches.dropna(subset=["winner"])
    valid = valid[valid["winner"] != "No Result"]

    X_encoded, feature_names = prepare_features(valid, scale_numerical=False)
    y_clf = valid.loc[X_encoded.index, "team1_won"]

    X_train, X_test, y_train, y_test = split_data(X_encoded, y_clf)

    # Classification
    clf = build_classifier()
    clf.fit(X_train, y_train)
    cv_clf = cross_validate_model(clf, X_encoded, y_clf, cv=5, scoring="accuracy")
    logger.info("RF Classifier CV accuracy: %.4f (+/- %.4f)", cv_clf["mean"], cv_clf["std"])

    # Regression: predict run margin
    runs_mask = valid["result"] == "runs"
    if runs_mask.sum() > 50:
        X_reg = X_encoded.loc[runs_mask.reindex(X_encoded.index, fill_value=False)]
        y_reg = valid.loc[X_reg.index, "result_margin"]

        reg = build_regressor()
        reg.fit(X_reg, y_reg)
        cv_reg = cross_validate_model(
            reg, X_reg, y_reg, cv=5, scoring="neg_mean_absolute_error"
        )
        logger.info("GBR Regressor CV MAE: %.4f", -cv_reg["mean"])
    else:
        reg = None
        logger.warning("Not enough 'runs' results for regression (%d)", runs_mask.sum())

    return clf, reg, X_test, y_test, X_encoded, y_clf, feature_names


def stage_evaluate(
    clf, reg, X_test, y_test, X_encoded, y_clf, feature_names
) -> dict:
    """Stage 7: Model evaluation and reporting."""
    logger.info("Evaluating models...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Classification metrics
    clf_metrics = evaluate_model(clf, X_test, y_test, task="classification")

    # Feature importances
    fi = get_feature_importances(clf, feature_names, top_n=15)

    results = {
        "classification": clf_metrics,
        "feature_importances": fi,
    }

    # Regression metrics (if regressor was trained)
    if reg is not None:
        # Use same split logic for regression evaluation
        reg_metrics = {
            "note": "Regression evaluated during training via cross-validation"
        }
        results["regression"] = reg_metrics

    # Generate and save report
    report = generate_report(results)
    report_path = RESULTS_DIR / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved evaluation report: %s", report_path)

    print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(stage: str | None = None) -> None:
    """Execute the full ML pipeline or a specific stage.

    Parameters
    ----------
    stage : str or None
        If None, run all stages sequentially. Otherwise, run only
        the specified stage: 'clean', 'eda', 'features', 'hypothesis',
        'train', 'evaluate'.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("IPL Match Predictor - ML Pipeline")
    logger.info("=" * 60)

    # Always need data
    matches, deliveries = stage_load()
    matches, deliveries = stage_clean(matches, deliveries)

    if stage == "clean":
        logger.info("Stage 'clean' complete.")
        return

    if stage is None or stage == "eda":
        stage_eda(matches, deliveries)
        if stage == "eda":
            return

    if stage is None or stage == "features":
        matches = stage_features(matches)
        if stage == "features":
            return

    if stage is None or stage == "hypothesis":
        stage_hypothesis(matches)
        if stage == "hypothesis":
            return

    if stage is None or stage == "train":
        clf, reg, X_test, y_test, X_enc, y_clf, feat_names = stage_train(matches)
        if stage == "train":
            return

    if stage is None or stage == "evaluate":
        if stage == "evaluate":
            # Need to train first
            matches = stage_features(matches)
            clf, reg, X_test, y_test, X_enc, y_clf, feat_names = stage_train(matches)
        stage_evaluate(clf, reg, X_test, y_test, X_enc, y_clf, feat_names)

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="IPL Match Predictor - End-to-end ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full pipeline
  python main.py --stage clean      Run data cleaning only
  python main.py --stage eda        Run EDA and save charts
  python main.py --stage features   Run feature engineering
  python main.py --stage hypothesis Run statistical tests
  python main.py --stage train      Train models
  python main.py --stage evaluate   Evaluate and generate report
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["clean", "eda", "features", "hypothesis", "train", "evaluate"],
        default=None,
        help="Run a specific pipeline stage (default: run all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        run_pipeline(stage=args.stage)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()