"""
Credit Risk Pipeline - End-to-End ML Pipeline

Usage:
    python main.py                     # Run full pipeline
    python main.py --stage clean       # Run only data cleaning
    python main.py --stage eda         # Run only EDA
    python main.py --stage features    # Run only feature engineering
    python main.py --stage train       # Run only model training
    python main.py --stage evaluate    # Run only evaluation
    python main.py --verbose           # Enable debug logging

Stages (run in order):
    clean      Load raw data and apply cleaning
    eda        Exploratory data analysis with charts
    features   Engineer domain features (DTI, utilization, loan burden)
    train      Train LR + GBC pipelines with cross-validation
    evaluate   Cost-sensitive threshold tuning, confusion matrix, reports
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_clean(cfg: dict) -> "pd.DataFrame":
    """Load and clean credit risk data."""
    from src.data_loader import load_data, clean_data

    logger.info("Stage: clean - loading and cleaning data")
    df = load_data(cfg)
    df = clean_data(df, cfg)
    logger.info(f"[OK] Cleaned data: {len(df)} rows, {df.shape[1]} columns")
    print(f"[OK] Cleaned data: {len(df)} rows, {df.shape[1]} columns")
    print(f"     Target distribution: {df['target'].value_counts().to_dict()}")
    return df


def stage_eda(df: "pd.DataFrame", cfg: dict) -> None:
    """Run exploratory data analysis and generate charts."""
    from src.eda import run_eda

    logger.info("Stage: eda - exploratory data analysis")
    run_eda(df, cfg)
    print("[OK] EDA charts generated")


def stage_features(df: "pd.DataFrame", cfg: dict) -> "pd.DataFrame":
    """Engineer domain features."""
    from src.features import engineer_features

    logger.info("Stage: features - engineering domain features")
    df = engineer_features(df, cfg)
    new_cols = [c for c in df.columns if c.startswith(("dti_", "utilization_", "log_", "age_"))]
    print(f"[OK] Engineered {len(new_cols)} new features: {new_cols}")
    return df


def stage_train(df: "pd.DataFrame", cfg: dict) -> dict:
    """Train models with cross-validation."""
    from src.models import train_and_evaluate

    logger.info("Stage: train - training models")
    results = train_and_evaluate(df, cfg)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  ROC-AUC: {metrics['roc_auc_mean']:.3f} (+/- {metrics['roc_auc_std']:.3f})")
        print(f"  Recall:  {metrics['recall_mean']:.3f}")
        print(f"  F1:      {metrics['f1_mean']:.3f}")

    return results


def stage_evaluate(df: "pd.DataFrame", results: dict, cfg: dict) -> None:
    """Cost-sensitive evaluation and reporting."""
    from src.evaluate import full_evaluation

    logger.info("Stage: evaluate - cost-sensitive evaluation")
    report = full_evaluation(df, results, cfg)
    print(f"\n[OK] Evaluation complete")
    print(f"     Optimal threshold: {report['optimal_threshold']:.2f}")
    print(f"     Minimum cost: ${report['min_cost']:,.0f}")

    # Save report
    report_path = PROJECT_ROOT / "artifacts" / "results" / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report["markdown"], encoding="utf-8")
    print(f"     Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(stage: str | None = None) -> None:
    """Run the full pipeline or a specific stage."""
    from src.data_loader import load_config

    cfg = load_config()

    # Stage 1: Clean
    df = stage_clean(cfg)
    if stage == "clean":
        return

    # Stage 2: EDA
    stage_eda(df, cfg)
    if stage == "eda":
        return

    # Stage 3: Features
    df = stage_features(df, cfg)
    if stage == "features":
        return

    # Stage 4: Train
    results = stage_train(df, cfg)
    if stage == "train":
        return

    # Stage 5: Evaluate
    stage_evaluate(df, results, cfg)
    if stage == "evaluate":
        return

    print("\n" + "=" * 60)
    print("[OK] Full pipeline completed successfully!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Pipeline - End-to-End ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["clean", "eda", "features", "train", "evaluate"],
        default=None,
        help="Run a specific pipeline stage (default: run all stages)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()
    try:
        run_pipeline(stage=args.stage)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline stopped by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed")
        print(f"\n[FAIL] {exc}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()