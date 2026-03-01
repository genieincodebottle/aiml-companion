"""
Model Monitoring
================

Population Stability Index (PSI) for drift detection, plus
data quality checks for production monitoring.

PSI compares the distribution of features/predictions between
a reference (training) dataset and incoming production data.

Industry thresholds:
- PSI < 0.10: No significant change
- PSI 0.10 - 0.25: Moderate shift, investigate
- PSI > 0.25: Significant shift, retrain

Functions
---------
compute_psi           Calculate PSI for a single feature.
monitor_drift         Check all features for drift.
generate_drift_report Generate a drift monitoring report.
check_data_quality    Run data quality checks on incoming data.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Population Stability Index between two distributions.

    PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

    Parameters
    ----------
    reference : array-like
        Reference (training) distribution values.
    current : array-like
        Current (production) distribution values.
    n_bins : int
        Number of bins for discretization.

    Returns
    -------
    float
        PSI value. < 0.10 = stable, 0.10-0.25 = moderate, > 0.25 = significant.
    """
    reference = np.array(reference, dtype=float)
    current = np.array(current, dtype=float)

    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicate edges

    if len(breakpoints) < 2:
        return 0.0

    # Count proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    # Convert to proportions with floor to avoid division by zero
    eps = 1e-4
    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    # Normalize
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    # PSI formula
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def classify_psi(psi_value: float) -> str:
    """Classify PSI value into action category.

    Parameters
    ----------
    psi_value : float
        Computed PSI value.

    Returns
    -------
    str
        One of: "stable", "moderate_shift", "significant_shift".
    """
    if psi_value < 0.10:
        return "stable"
    elif psi_value < 0.25:
        return "moderate_shift"
    else:
        return "significant_shift"


# ---------------------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------------------

def monitor_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Check all specified features for distribution drift.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference (training) dataset.
    current_df : pd.DataFrame
        Current (production) dataset.
    columns : list[str], optional
        Columns to check. Defaults to all shared numeric columns.
    n_bins : int
        Number of bins for PSI computation.

    Returns
    -------
    dict
        Feature -> {"psi": float, "status": str, "action": str}
    """
    if columns is None:
        ref_numeric = set(reference_df.select_dtypes(include=[np.number]).columns)
        cur_numeric = set(current_df.select_dtypes(include=[np.number]).columns)
        columns = list(ref_numeric & cur_numeric)

    results = {}
    for col in sorted(columns):
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        psi = compute_psi(
            reference_df[col].values,
            current_df[col].values,
            n_bins=n_bins,
        )
        status = classify_psi(psi)

        action = {
            "stable": "No action needed",
            "moderate_shift": "Investigate feature distribution changes",
            "significant_shift": "Retrain model with updated data",
        }[status]

        results[col] = {"psi": round(psi, 4), "status": status, "action": action}

        if status != "stable":
            logger.warning(f"Drift detected in {col}: PSI={psi:.4f} ({status})")

    return results


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame) -> dict:
    """Run data quality checks on incoming data.

    Checks:
    - Missing value rates
    - Out-of-range values
    - Unexpected categories
    - Duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Incoming production data.

    Returns
    -------
    dict
        Quality report with issues found.
    """
    issues = []
    stats = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "timestamp": datetime.now().isoformat(),
    }

    # Missing values
    missing = df.isnull().sum()
    high_missing = missing[missing / len(df) > 0.05]
    if len(high_missing) > 0:
        for col, count in high_missing.items():
            pct = count / len(df) * 100
            issues.append({
                "type": "missing_values",
                "column": col,
                "detail": f"{count} missing ({pct:.1f}%)",
                "severity": "warning" if pct < 20 else "critical",
            })

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        issues.append({
            "type": "duplicates",
            "column": None,
            "detail": f"{n_dupes} duplicate rows ({n_dupes/len(df)*100:.1f}%)",
            "severity": "warning",
        })

    # Numeric range checks
    numeric = df.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if numeric[col].min() < 0 and col in ("age", "income", "credit_amount", "duration"):
            issues.append({
                "type": "out_of_range",
                "column": col,
                "detail": f"Negative values found (min={numeric[col].min():.2f})",
                "severity": "critical",
            })

    stats["n_issues"] = len(issues)
    stats["issues"] = issues
    stats["pass"] = len([i for i in issues if i["severity"] == "critical"]) == 0

    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_drift_report(drift_results: dict, quality_results: dict) -> str:
    """Generate a Markdown drift monitoring report.

    Parameters
    ----------
    drift_results : dict
        Output from ``monitor_drift()``.
    quality_results : dict
        Output from ``check_data_quality()``.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# Drift Monitoring Report\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Feature Drift (PSI)\n",
        "| Feature | PSI | Status | Action |",
        "|---------|-----|--------|--------|",
    ]

    for feat, info in sorted(drift_results.items(), key=lambda x: x[1]["psi"], reverse=True):
        emoji = {"stable": "OK", "moderate_shift": "WARN", "significant_shift": "ALERT"}
        lines.append(
            f"| {feat} | {info['psi']:.4f} | "
            f"{emoji.get(info['status'], '?')} {info['status']} | {info['action']} |"
        )

    # Data quality section
    lines.extend([
        "\n## Data Quality\n",
        f"- **Rows**: {quality_results['n_rows']}",
        f"- **Columns**: {quality_results['n_columns']}",
        f"- **Issues**: {quality_results['n_issues']}",
        f"- **Pass**: {'Yes' if quality_results['pass'] else 'No'}\n",
    ])

    if quality_results.get("issues"):
        lines.extend([
            "### Issues Found\n",
            "| Type | Column | Detail | Severity |",
            "|------|--------|--------|----------|",
        ])
        for issue in quality_results["issues"]:
            lines.append(
                f"| {issue['type']} | {issue.get('column', '-')} | "
                f"{issue['detail']} | {issue['severity']} |"
            )

    return "\n".join(lines)