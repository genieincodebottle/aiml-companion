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
from sklearn.metrics import accuracy_score, f1_score

from src.models import (
    build_classifier,
    build_ensemble,
    ensemble_predict_proba,
    build_regressor,
    build_xgb_classifier,
    build_neural_classifier,
    cross_validate_model,
    forward_cv,
    evaluate_model,
    get_feature_importances,
    prepare_features,
    split_data,
    split_data_by_season,
    majority_class_baseline,
    _HAS_XGBOOST,
    _HAS_TORCH,
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
    use_neural_net: bool = False,
) -> dict:
    """Stage 6: Model training with cross-validation.

    Trains Random Forest and XGBoost, optionally a Neural Net, plus the
    calibrated 4-model ensemble and one regressor (Gradient Boosting).
    Returns a dict with all trained models and data splits.

    Model selection (the "CV pick") uses ONLY the training seasons, with
    forward-in-time folds (``forward_cv()``, an expanding window on the
    date-sorted rows). The test seasons are never seen by selection.

    ``use_neural_net`` is off by default so the result does not depend on
    whether the optional ``torch`` package happens to be installed. Turn it
    on with ``python main.py --neural-net`` (needs torch).
    """
    logger.info("Training models...")

    # Classification: predict match winner (team1 wins = 1)
    matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)
    valid = matches.dropna(subset=["winner"])
    valid = valid[valid["winner"] != "No Result"]

    X_encoded, feature_names = prepare_features(valid, scale_numerical=False)
    y_clf = valid.loc[X_encoded.index, "team1_won"]

    # Out-of-time split: train on earlier seasons, test on the most recent.
    #
    # This project predicts matches that have not been played, so the test set
    # has to sit in the future of the training set. A random split trains on
    # 2025 to predict 2024 -- an easier problem than the real one, and it read
    # 0.5150 where the honest 2025 holdout reads 0.4507.
    seasons = pd.to_datetime(valid.loc[X_encoded.index, "date"]).dt.year
    try:
        X_train, X_test, y_train, y_test = split_data_by_season(
            X_encoded, y_clf, seasons, holdout_seasons=2)
        baseline = majority_class_baseline(y_train, y_test)
        logger.info(
            "Out-of-time split: train=%d (through %d), test=%d (%d onwards). "
            "Majority-class baseline on the test seasons: %.4f",
            len(y_train), int(seasons.max()) - 2, len(y_test),
            int(seasons.max()) - 1, baseline)
    except ValueError as e:
        logger.warning("Falling back to a random split: %s", e)
        X_train, X_test, y_train, y_test = split_data(X_encoded, y_clf)
        baseline = majority_class_baseline(y_train, y_test)
        seasons = None

    # Selection CV runs on the training rows only. Forward-in-time folds need
    # date order; the random-split fallback has none, so it keeps plain 5-fold.
    if seasons is not None:
        train_seasons = seasons.iloc[: len(y_train)].to_numpy()
        assert (np.diff(train_seasons) >= 0).all(), "training rows must be date-sorted"
        cv_splitter = forward_cv(5)
        for i, (tr, te) in enumerate(cv_splitter.split(X_train), 1):
            logger.info(
                "CV fold %d: train %d rows (%d-%d), validate %d rows (%d-%d)",
                i, len(tr), train_seasons[tr].min(), train_seasons[tr].max(),
                len(te), train_seasons[te].min(), train_seasons[te].max())
    else:
        cv_splitter = 5

    classifiers = {}
    cv_results = {}

    # 1. Random Forest
    clf_rf = build_classifier()
    clf_rf.fit(X_train, y_train)
    cv_rf = cross_validate_model(clf_rf, X_train, y_train, cv=cv_splitter, scoring="accuracy")
    classifiers["Random Forest"] = clf_rf
    cv_results["Random Forest"] = cv_rf
    logger.info("RF Classifier CV accuracy: %.4f (+/- %.4f)", cv_rf["mean"], cv_rf["std"])

    # 2. XGBoost
    if _HAS_XGBOOST:
        clf_xgb = build_xgb_classifier()
        clf_xgb.fit(X_train, y_train)
        cv_xgb = cross_validate_model(clf_xgb, X_train, y_train, cv=cv_splitter, scoring="accuracy")
        classifiers["XGBoost"] = clf_xgb
        cv_results["XGBoost"] = cv_xgb
        logger.info("XGBoost CV accuracy: %.4f (+/- %.4f)", cv_xgb["mean"], cv_xgb["std"])
    else:
        logger.warning("xgboost not installed - skipping XGBoost classifier")

    # 3. Neural Network (opt-in: torch is an optional dependency)
    if use_neural_net and _HAS_TORCH:
        clf_nn = build_neural_classifier()
        clf_nn.fit(X_train, y_train)
        cv_nn = cross_validate_model(clf_nn, X_train, y_train, cv=cv_splitter, scoring="accuracy")
        classifiers["Neural Net"] = clf_nn
        cv_results["Neural Net"] = cv_nn
        logger.info("Neural Net CV accuracy: %.4f (+/- %.4f)", cv_nn["mean"], cv_nn["std"])
    elif use_neural_net:
        logger.warning("torch not installed - skipping Neural Net classifier")

    # 4. The calibrated 4-model weighted ensemble (RF + XGB + GB + LR) that
    # src/predict.py and scripts/predict_all_2026.py actually use. It is
    # scored on the same holdout but is not part of the CV pick above.
    ensemble_metrics = None
    if _HAS_XGBOOST:
        # Same recency weighting src/predict.py trains the deployed ensemble with.
        ens_sw = valid.loc[X_train.index, "recency_weight"].to_numpy()
        ens = build_ensemble(X_train, y_train, sample_weight=ens_sw, calibrate=True)
        ens_pred = (ensemble_predict_proba(ens, X_test) >= 0.5).astype(int)
        ensemble_metrics = {
            "accuracy": float(accuracy_score(y_test, ens_pred)),
            "f1": float(f1_score(y_test, ens_pred, average="weighted")),
        }
        logger.info("4-model ensemble test accuracy: %.4f", ensemble_metrics["accuracy"])

    # Regression: predict run margin
    reg = None
    runs_mask = valid["result"] == "runs"
    if runs_mask.sum() > 50:
        # Training seasons only, same forward-in-time folds as the classifiers.
        X_reg = X_train.loc[runs_mask.reindex(X_train.index, fill_value=False)]
        y_reg = valid.loc[X_reg.index, "result_margin"]

        reg = build_regressor()
        reg.fit(X_reg, y_reg)
        reg_cv = forward_cv(5) if seasons is not None else 5
        cv_reg = cross_validate_model(
            reg, X_reg, y_reg, cv=reg_cv, scoring="neg_mean_absolute_error"
        )
        logger.info("GBR Regressor CV MAE: %.4f", -cv_reg["mean"])
    else:
        logger.warning("Not enough 'runs' results for regression (%d)", runs_mask.sum())

    # Print comparison table
    logger.info("")
    logger.info("=" * 50)
    logger.info("  MODEL COMPARISON (forward CV, training seasons)")
    logger.info("=" * 50)
    for name, cv in sorted(cv_results.items(), key=lambda x: -x[1]["mean"]):
        logger.info("  %-15s  %.4f (+/- %.4f)", name, cv["mean"], cv["std"])
    logger.info("=" * 50)

    # Pick the best classifier
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    logger.info("Best classifier: %s (%.4f)", best_name, cv_results[best_name]["mean"])

    return {
        "classifiers": classifiers,
        "cv_results": cv_results,
        "best_name": best_name,
        "best_clf": classifiers[best_name],
        "reg": reg,
        "ensemble_metrics": ensemble_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "X_encoded": X_encoded,
        "y_clf": y_clf,
        "feature_names": feature_names,
        # The number every accuracy below has to be read against.
        "baseline_accuracy": baseline,
    }


def stage_evaluate(train_output: dict) -> dict:
    """Stage 7: Model evaluation and reporting.

    Evaluates all trained classifiers and picks the best one
    based on test-set accuracy. Generates a comparison report.
    """
    logger.info("Evaluating models...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    classifiers = train_output["classifiers"]
    X_test = train_output["X_test"]
    y_test = train_output["y_test"]
    feature_names = train_output["feature_names"]
    reg = train_output["reg"]

    all_metrics = {}
    for name, clf in classifiers.items():
        metrics = evaluate_model(clf, X_test, y_test, task="classification")
        all_metrics[name] = metrics
        logger.info(
            "%s test accuracy: %.4f  F1: %.4f",
            name,
            metrics["accuracy"],
            metrics["f1"],
        )

    if train_output.get("ensemble_metrics"):
        all_metrics["4-model ensemble"] = train_output["ensemble_metrics"]
        logger.info("4-model ensemble test accuracy: %.4f",
                    train_output["ensemble_metrics"]["accuracy"])

    # Feature importances from tree-based models
    best_name = train_output["best_name"]
    best_clf = train_output["best_clf"]

    fi = None
    fi_source = best_name
    try:
        fi = get_feature_importances(best_clf, feature_names, top_n=15)
    except AttributeError:
        # Neural Net doesn't expose feature_importances_
        # Fall back to Random Forest if available
        if "Random Forest" in classifiers:
            fi = get_feature_importances(
                classifiers["Random Forest"], feature_names, top_n=15
            )
            fi_source = "Random Forest (the best model has none)"
            logger.info("Feature importances from Random Forest (best model has none)")

    results = {
        # Carry the baseline into the report: accuracy on a near-balanced
        # binary problem is unreadable without it.
        "classification": {**all_metrics[best_name],
                           "baseline_accuracy": train_output.get("baseline_accuracy")},
        "all_classifiers": {
            name: {"accuracy": m["accuracy"], "f1": m["f1"]}
            for name, m in all_metrics.items()
        },
        "best_model": best_name,
        "cv_results": train_output["cv_results"],
    }

    if fi is not None:
        results["feature_importances"] = fi
        results["feature_importances_source"] = fi_source

    if reg is not None:
        results["regression"] = {
            "note": "Regression evaluated during training via cross-validation"
        }

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
def run_pipeline(stage: str | None = None, use_neural_net: bool = False) -> None:
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

    # The train stage needs the engineered features too (Elo, momentum, H2H),
    # otherwise it silently trains on team/venue one-hots alone.
    if stage in (None, "features", "train"):
        matches = stage_features(matches)
        if stage == "features":
            return

    if stage is None or stage == "hypothesis":
        stage_hypothesis(matches)
        if stage == "hypothesis":
            return

    if stage is None or stage == "train":
        train_output = stage_train(matches, use_neural_net=use_neural_net)
        if stage == "train":
            return

    if stage is None or stage == "evaluate":
        if stage == "evaluate":
            # Need to train first
            matches = stage_features(matches)
            train_output = stage_train(matches, use_neural_net=use_neural_net)
        stage_evaluate(train_output)

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
        "--neural-net",
        action="store_true",
        help="Also train/compare the PyTorch Neural Net (needs the optional "
             "torch package). Off by default so results do not depend on "
             "what happens to be installed.",
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
        run_pipeline(stage=args.stage, use_neural_net=args.neural_net)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()