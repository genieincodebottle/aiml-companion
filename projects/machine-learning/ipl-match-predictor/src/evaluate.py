"""
evaluate.py - Model Evaluation and Reporting Utilities
========================================================

Generates structured evaluation reports in Markdown format
and prints summary tables to the console.

Usage
-----
>>> from src.evaluate import generate_report, print_summary
>>> report = generate_report(results)
>>> print_summary(results)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def generate_report(results: Dict[str, Any]) -> str:
    """Generate a Markdown evaluation report from model results.

    Parameters
    ----------
    results : dict
        Dictionary containing 'classification' and optionally
        'regression' and 'feature_importances' keys.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines = [
        "# IPL Match Predictor - Evaluation Report",
        "",
        "## Classification: Match Winner Prediction",
        "",
    ]

    clf = results.get("classification", {})
    if clf:
        acc = clf.get("accuracy", 0)
        baseline = clf.get("baseline_accuracy")
        best_model = results.get("best_model")
        if best_model:
            lines.append(f"- **Model (best forward-CV accuracy, training seasons)**: {best_model}")
        lines.extend([
            f"- **Accuracy**: {acc:.4f}",
            f"- **F1 Score (weighted)**: {clf.get('f1', 0):.4f}",
        ])
        if baseline is not None:
            # Accuracy alone is unreadable on a near-balanced binary problem.
            # Always predicting the same team scores ~0.52-0.54 here, so a
            # model at 0.53 has demonstrated nothing at all -- and printing the
            # two numbers next to each other is the difference between a
            # result and a coin flip wearing one.
            lift = acc - baseline
            # A holdout this small cannot resolve a lift of a few points:
            # one standard error of an accuracy near 0.5 is
            # sqrt(0.25 / n), about 0.042 for n = 142. Only call it a win
            # when the lift clears two standard errors.
            cm_ = clf.get("confusion_matrix")
            n_test = int(np.sum(cm_)) if cm_ is not None else 0
            se = float(np.sqrt(0.25 / n_test)) if n_test else None
            if se is not None and lift > 2 * se:
                verdict = "BEATS the baseline"
            elif lift > 0:
                verdict = "is ahead of the baseline, but within sampling noise"
            else:
                verdict = "does NOT beat the baseline"
            lines.extend([
                f"- **Majority-class baseline**: {baseline:.4f}",
                f"- **Lift over baseline**: {lift:+.4f} -- the model {verdict}",
            ])
            if se is not None:
                lines.append(
                    f"- **Test matches**: {n_test} "
                    f"(one standard error of the accuracy is about {se:.3f})"
                )
        lines.extend([
            "",
            "> Evaluated on an **out-of-time** split: trained on earlier "
            "seasons, tested on the most recent two. A random split reads "
            "higher (the Random Forest scores ~0.53 on a random 25% holdout) "
            "because it lets the model train on 2025 to predict 2024, which "
            "is not a thing deployment can do.",
            "",
        ])

        # Every classifier on the same holdout, so the reader can see that the
        # CV-picked one is not special.
        allc = results.get("all_classifiers")
        if allc:
            cv = results.get("cv_results", {})
            lines.extend([
                "### All classifiers on the same holdout",
                "",
                "| Model | Test accuracy | F1 (weighted) | Forward CV accuracy (train seasons) |",
                "|-------|---------------|---------------|-----------------------------------|",
            ])
            for name, m in allc.items():
                cv_mean = cv.get(name, {}).get("mean")
                cv_txt = f"{cv_mean:.4f}" if cv_mean == cv_mean and cv_mean is not None else "n/a"
                lines.append(
                    f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} | {cv_txt} |"
                )
            lines.append("")

        # Confusion matrix
        cm = clf.get("confusion_matrix")
        if cm is not None:
            lines.extend([
                "### Confusion Matrix",
                "",
                "```",
                "                 Predicted",
                "              Team2  Team1",
                f"Actual Team2   {cm[0][0]:4d}   {cm[0][1]:4d}",
                f"Actual Team1   {cm[1][0]:4d}   {cm[1][1]:4d}",
                "```",
                "",
            ])

        # Classification report
        report = clf.get("report", {})
        if report:
            lines.extend([
                "### Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 |",
                "|-------|-----------|--------|-----|",
            ])
            for cls_name in ["Team 2 Wins/Other", "Team 1 Wins"]:
                cls_metrics = report.get(cls_name, {})
                if cls_metrics:
                    lines.append(
                        f"| {cls_name} | "
                        f"{cls_metrics.get('precision', 0):.3f} | "
                        f"{cls_metrics.get('recall', 0):.3f} | "
                        f"{cls_metrics.get('f1-score', 0):.3f} |"
                    )
            lines.append("")

    # Feature importances
    fi = results.get("feature_importances")
    if fi is not None and len(fi) > 0:
        lines.extend([
            "## Top Feature Importances",
            "",
            f"Source model: {results.get('feature_importances_source', 'best classifier')}",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ])
        for i, row in fi.head(10).iterrows():
            lines.append(f"| {i + 1} | {row['Feature']} | {row['Importance']:.4f} |")
        lines.append("")

    # Regression
    reg = results.get("regression", {})
    if reg and "note" not in reg:
        lines.extend([
            "## Regression: Run Margin Prediction",
            "",
            f"- **MAE**: {reg.get('mae', 0):.4f}",
            f"- **RMSE**: {reg.get('rmse', 0):.4f}",
            f"- **R2 Score**: {reg.get('r2', 0):.4f}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Report generated by IPL Match Predictor pipeline*",
    ])

    return "\n".join(lines)


def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise summary of model evaluation results.

    Parameters
    ----------
    results : dict
        Dictionary containing model evaluation metrics.
    """
    print("\n" + "=" * 50)
    print("  IPL Match Predictor - Results Summary")
    print("=" * 50)

    clf = results.get("classification", {})
    if clf:
        print(f"\n  Classification (Match Winner)")
        print(f"  {'Accuracy:':<20} {clf.get('accuracy', 0):.4f}")
        print(f"  {'F1 (weighted):':<20} {clf.get('f1', 0):.4f}")

    fi = results.get("feature_importances")
    if fi is not None and len(fi) > 0:
        print(f"\n  Top 5 Features:")
        for i, row in fi.head(5).iterrows():
            print(f"    {i + 1}. {row['Feature']:<30} {row['Importance']:.4f}")

    reg = results.get("regression", {})
    if reg and "note" not in reg:
        print(f"\n  Regression (Run Margin)")
        print(f"  {'MAE:':<20} {reg.get('mae', 0):.4f}")
        print(f"  {'RMSE:':<20} {reg.get('rmse', 0):.4f}")
        print(f"  {'R2:':<20} {reg.get('r2', 0):.4f}")

    print("\n" + "=" * 50)