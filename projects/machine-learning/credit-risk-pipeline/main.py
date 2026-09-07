"""
Credit Risk Pipeline - End-to-End ML Pipeline

Usage:
    python main.py                     # Run full pipeline
    python main.py --stage clean       # Run only data cleaning
    python main.py --stage eda         # Run only EDA
    python main.py --stage features    # Run only feature engineering
    python main.py --stage train       # Run only model training
    python main.py --stage evaluate    # Run only evaluation
    python main.py --stage serve       # Start the FastAPI prediction server
    python main.py --verbose           # Enable debug logging

Stages (run in order):
    clean      Load raw data and apply cleaning
    eda        Exploratory data analysis with charts
    features   Engineer domain features (DTI, utilization, loan burden)
    train      Train LR + GBC pipelines with cross-validation
    evaluate   Cost-sensitive threshold tuning, confusion matrix, reports
    serve      Launch the FastAPI server (requires a trained model)
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
    # Diff the actual columns. This used to filter on a hardcoded prefix list
    # -- ("dti_", "utilization_", "log_", "age_") -- which did not include
    # `loan_burden`, so the engineered feature that ends up THIRD by SHAP
    # importance was missing from the count. The line read "Engineered 1 new
    # features" while the model was using two, and it also never mentioned the
    # protected column being dropped. Compare the frames instead of guessing
    # at names.
    before = list(df.columns)
    df = engineer_features(df, cfg)
    after = list(df.columns)

    added = [c for c in after if c not in before]
    removed = [c for c in before if c not in after]

    print(f"[OK] Engineered {len(added)} new features: {added}")
    if removed:
        print(f"     Dropped {len(removed)} protected/unused columns: {removed}")
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
    if report.get("shap_error"):
        # SHAP is the adverse-action half of this project. It failed
        # silently for a long time: caught, logged at WARNING, and the
        # run still ended with "completed successfully" while a stale
        # shap_importance.png sat in artifacts looking current.
        print("[WARN] SHAP explanations NOT computed: "
              + report["shap_error"])
        print("       No adverse-action reasons in this report.")

    print(f"\n[OK] Evaluation complete")
    print(f"     Optimal threshold: {report['optimal_threshold']:.2f}")
    print(f"     Minimum cost: ${report['min_cost']:,.0f}")

    # Save report
    report_path = PROJECT_ROOT / "artifacts" / "results" / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report["markdown"], encoding="utf-8")
    print(f"     Report saved to: {report_path}")

    # Save tuned threshold for the serving API (src/serve.py reads this)
    import json

    threshold_path = PROJECT_ROOT / "artifacts" / "results" / "threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "threshold": round(float(report["optimal_threshold"]), 4),
                "min_cost": float(report["min_cost"]),
                "best_model": report["best_model"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"     Threshold saved to: {threshold_path}")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def stage_serve() -> None:
    """Launch the FastAPI prediction server (blocks until stopped)."""
    import uvicorn

    model_path = PROJECT_ROOT / "artifacts" / "results" / "best_model.joblib"
    if not model_path.exists():
        print(f"[FAIL] No trained model at {model_path}. Run `python main.py` first.")
        sys.exit(1)

    print("Starting API server at http://localhost:8000 (Ctrl+C to stop)")
    uvicorn.run("src.serve:app", host="127.0.0.1", port=8000)


def run_pipeline(stage: str | None = None) -> None:
    """Run the full pipeline or a specific stage."""
    from src.data_loader import load_config

    # Serve does not need the data pipeline
    if stage == "serve":
        stage_serve()
        return

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
        choices=["clean", "eda", "features", "train", "evaluate", "serve"],
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
    # Third-party debug output drowns out pipeline logs under --verbose
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

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