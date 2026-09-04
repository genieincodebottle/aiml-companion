"""
Evaluation & Reporting
======================

Cost-sensitive evaluation with threshold tuning, confusion matrix
analysis, SHAP explanations, and Markdown report generation.

The cost matrix uses a 10:1 FN/FP ratio reflecting the asymmetric
cost of missing a defaulter vs. denying a good borrower.

Functions
---------
full_evaluation         Run complete evaluation pipeline.
find_optimal_threshold  Cost-sensitive threshold optimization.
generate_report         Generate Markdown evaluation report.
plot_precision_recall   Plot PR curve with optimal threshold.
plot_confusion_matrix   Plot confusion matrix heatmap.
compute_shap_values     Compute SHAP feature importances.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def full_evaluation(df: pd.DataFrame, results: dict, cfg: dict) -> dict:
    """Run complete evaluation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data.
    results : dict
        Training results from ``models.train_and_evaluate()``.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Report with ``optimal_threshold``, ``min_cost``, ``markdown``.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    target_col = cfg.get("target_column", "target")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Use best model for evaluation
    best_name = max(results, key=lambda k: results[k]["roc_auc_mean"])
    pipeline = results[best_name]["pipeline"]

    # Out-of-fold probabilities: each row scored by a model that never trained
    # on it. Everything downstream -- the cost curve, the chosen threshold, the
    # confusion matrix, the classification report, the threshold handed to the
    # serving API -- depends on these being honest.
    #
    # This line used to read `pipeline.predict_proba(X)`, scoring the
    # deployment fit on its own training rows. See models.train_and_evaluate
    # for what that cost: a reported total cost of 1 against a real 712.
    y_proba = results[best_name].get("oof_proba")
    if y_proba is None:
        raise KeyError(
            f"{best_name} has no out-of-fold predictions. Evaluation must not "
            "fall back to in-sample scoring -- re-run training so "
            "train_and_evaluate() can produce them."
        )

    # Cost-sensitive threshold tuning
    cost_fn = cfg.get("cost_false_negative", 10)
    cost_fp = cfg.get("cost_false_positive", 1)
    optimal_threshold, min_cost = find_optimal_threshold(y, y_proba, cost_fn, cost_fp)

    # Predictions at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    # Plots
    plot_precision_recall(y, y_proba, optimal_threshold)
    plot_confusion_matrix(y, y_pred, best_name)

    # SHAP. Optional in the sense that a missing `shap` install must not take
    # the pipeline down -- but NOT optional in the sense of "quietly absent".
    # Adverse-action reasons are the regulatory point of this project, so a
    # failure here is carried into the report and surfaced by the caller
    # instead of vanishing into a log line nobody reads.
    shap_summary = None
    shap_error = None
    try:
        shap_summary = compute_shap_values(pipeline, X, cfg)
    except Exception as e:
        shap_error = f"{type(e).__name__}: {e}"
        logger.warning(f"SHAP analysis FAILED: {shap_error}")

    # Classification report
    cls_report = classification_report(y, y_pred, output_dict=True)

    # Generate Markdown report
    markdown = generate_report(
        results=results,
        best_name=best_name,
        optimal_threshold=optimal_threshold,
        min_cost=min_cost,
        cls_report=cls_report,
        shap_summary=shap_summary,
        cfg=cfg,
        shap_error=shap_error,
    )

    return {
        "optimal_threshold": optimal_threshold,
        "min_cost": min_cost,
        "classification_report": cls_report,
        "best_model": best_name,
        "markdown": markdown,
        "shap_error": shap_error,
    }


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10,
    cost_fp: float = 1,
) -> tuple[float, float]:
    """Find threshold that minimizes total misclassification cost.

    Cost = FN * cost_fn + FP * cost_fp

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    cost_fn : float
        Cost per false negative (missed default).
    cost_fp : float
        Cost per false positive (denied good borrower).

    Returns
    -------
    tuple[float, float]
        (optimal_threshold, minimum_cost)
    """
    # The lower bound used to be 0.05, and on this dataset the search returned
    # exactly 0.05 -- the first value it was allowed to try. That is a search
    # hitting its own wall, not a minimum: cost at 0.01 is lower still. A grid
    # whose argmin sits on its boundary has not found an optimum, so the bound
    # starts at the bottom now and _warn_if_degenerate() below says so out loud
    # when the answer is still a corner.
    thresholds = np.arange(0.01, 0.95, 0.01)
    best_threshold = 0.5
    best_cost = float("inf")

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fn = cm[1, 0]  # Actual default, predicted no-default
        fp = cm[0, 1]  # Actual no-default, predicted default
        total_cost = fn * cost_fn + fp * cost_fp

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = thresh

    _warn_if_degenerate(y_true, y_proba, best_threshold, thresholds)

    logger.info(
        f"Optimal threshold: {best_threshold:.2f} "
        f"(cost: ${best_cost:,.0f}, FN cost={cost_fn}, FP cost={cost_fp})"
    )
    return best_threshold, best_cost


def _warn_if_degenerate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    grid: np.ndarray,
) -> None:
    """Say out loud when the cost-minimising threshold is not a usable one.

    A 10:1 FN/FP ratio with no constraint on approval rate has an obvious
    degenerate answer: flag nearly everybody. It minimises the stated cost and
    it is not a lending policy anybody could ship. The number still gets
    written to threshold.json and handed to the serving API, so the one thing
    that must not happen is for it to go out unremarked.
    """
    y_true = np.asarray(y_true)
    flagged = float((np.asarray(y_proba) >= threshold).mean())

    if threshold <= grid[0] + 1e-9:
        logger.warning(
            "Optimal threshold %.2f is the LOWEST value in the search grid. "
            "The cost curve may still be falling past the edge, so this is a "
            "boundary hit rather than a minimum.", threshold)

    if flagged >= 0.5:
        negatives = int((y_true == 0).sum())
        rejected_good = int(((np.asarray(y_proba) >= threshold) & (y_true == 0)).sum())
        logger.warning(
            "Threshold %.2f flags %.0f%% of all applicants and rejects %d of "
            "%d good borrowers. Cost-minimising, but not a shippable policy: a "
            "real deployment constrains approval rate or capital, and the "
            "10:1 cost ratio alone does not. Treat this number as the "
            "unconstrained corner it is.",
            threshold, flagged * 100, rejected_good, negatives)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_precision_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    optimal_threshold: float,
) -> None:
    """Plot Precision-Recall curve with optimal threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "b-", lw=2, label="PR Curve")

    # Mark optimal threshold
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.plot(
        recall[idx],
        precision[idx],
        "ro",
        markersize=12,
        label=f"Optimal (t={optimal_threshold:.2f})",
    )

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve with Optimal Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved precision_recall_curve.png")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Plot confusion matrix heatmap."""
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(pipeline: object, X: pd.DataFrame, cfg: dict) -> dict:
    """Compute SHAP feature importances for the fitted pipeline.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted pipeline with preprocessor + classifier.
    X : pd.DataFrame
        Feature matrix.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Top features with mean absolute SHAP values.
    """
    import shap

    # Get the classifier from the pipeline
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Transform features
    X_transformed = preprocessor.transform(X)

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()

    # Compute SHAP values (use sample for speed).
    #
    # This used to read `shap.Explainer(classifier, X_sample)` -- passing the
    # sample as a BACKGROUND dataset, which makes shap pick the interventional
    # perturbation path. On this gradient-boosted model that path failed its
    # own additivity check every run ("sum of the SHAP values was -2.239625,
    # while the model output was -2.379934"), the exception was caught below,
    # and the pipeline printed "completed successfully" with no SHAP at all.
    # The figure on disk was left over from an older run, so nothing looked
    # wrong. TreeExplainer with no background uses the exact path-dependent
    # algorithm, which is both faster and additive by construction.
    n_sample = min(len(X), 500)
    X_dense = X_transformed.toarray() if issparse(X_transformed) else np.asarray(X_transformed)
    X_sample = X_dense[:n_sample]

    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(explainer.shap_values(X_sample))

    # Binary classifiers may return (n, features, 2). Keep the positive class.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    # Mean absolute SHAP values
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(15, len(importance))
    top_features = list(importance.keys())[:top_n]
    top_values = [importance[f] for f in top_features]

    ax.barh(range(top_n), top_values[::-1], color="#3b82f6")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved shap_importance.png")

    return {"top_features": dict(list(importance.items())[:10])}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: dict,
    best_name: str,
    optimal_threshold: float,
    min_cost: float,
    cls_report: dict,
    shap_summary: dict | None,
    cfg: dict,
    shap_error: str | None = None,
) -> str:
    """Generate Markdown evaluation report.

    Parameters
    ----------
    results : dict
        Cross-validation results for all models.
    best_name : str
        Name of the best model.
    optimal_threshold : float
        Cost-optimal classification threshold.
    min_cost : float
        Minimum total misclassification cost.
    cls_report : dict
        sklearn classification report dict.
    shap_summary : dict or None
        SHAP feature importance summary.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Credit Risk Pipeline - Evaluation Report\n",
        "## Model Comparison\n",
        "| Model | ROC-AUC | Recall | F1 |",
        "|-------|---------|--------|----|",
    ]

    for name, metrics in results.items():
        marker = " **" if name == best_name else ""
        lines.append(
            f"| {name}{marker} | "
            f"{metrics['roc_auc_mean']:.3f} +/- {metrics['roc_auc_std']:.3f} | "
            f"{metrics['recall_mean']:.3f} | "
            f"{metrics['f1_mean']:.3f} |"
        )

    lines.extend([
        f"\n**Best Model**: {best_name}\n",
        "## Cost-Sensitive Threshold Tuning\n",
        f"- **Optimal Threshold**: {optimal_threshold:.2f}",
        f"- **Minimum Cost**: ${min_cost:,.0f}",
        f"- **Cost Ratio**: FN={cfg.get('cost_false_negative', 10)}x, FP={cfg.get('cost_false_positive', 1)}x\n",
        "## Classification Report (at optimal threshold)\n",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ])

    for label in ["0", "1"]:
        if label in cls_report:
            m = cls_report[label]
            lines.append(
                f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1-score']:.3f} | {m['support']:.0f} |"
            )

    if shap_error:
        lines += [
            "\n## Top Features (SHAP)\n",
            f"> **NOT COMPUTED.** SHAP failed with `{shap_error}`.",
            "> There are no adverse-action reasons in this report. Any",
            "> `shap_importance.png` in artifacts/figures is from an earlier",
            "> run and does not describe this model.\n",
        ]
    elif shap_summary and "top_features" in shap_summary:
        lines.extend([
            "\n## Top Features (SHAP)\n",
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|-------------|",
        ])
        for i, (feat, val) in enumerate(shap_summary["top_features"].items(), 1):
            lines.append(f"| {i} | {feat} | {val:.4f} |")

    lines.extend([
        "\n## Artifacts\n",
        "- `artifacts/figures/precision_recall_curve.png`",
        "- `artifacts/figures/confusion_matrix.png`",
        "- `artifacts/figures/shap_importance.png`",
        "- `artifacts/results/best_model.joblib`",
    ])

    return "\n".join(lines)