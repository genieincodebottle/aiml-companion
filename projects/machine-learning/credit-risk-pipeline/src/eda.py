"""
Exploratory Data Analysis
=========================

Generates EDA charts and summary statistics for the credit risk dataset.
All plots are saved to ``artifacts/figures/``.

Functions
---------
run_eda             Run full EDA suite.
plot_distributions  Plot feature distributions by target class.
plot_correlation    Plot correlation heatmap.
plot_target_rate    Plot default rate by categorical segments.
summarize_data      Print summary statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, cfg: dict) -> None:
    """Run full EDA suite and save charts.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with ``target`` column.
    cfg : dict
        Pipeline configuration.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summarize_data(df)
    plot_target_distribution(df, cfg)
    plot_distributions(df, cfg)
    plot_correlation(df, cfg)
    plot_target_rate(df, cfg)

    logger.info(f"EDA charts saved to {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_data(df: pd.DataFrame) -> None:
    """Print summary statistics to the logger."""
    logger.info("=" * 50)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    logger.info(f"Default rate: {df['target'].mean():.2%}")

    # Numeric summary
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        logger.info(f"\nNumeric features ({len(numeric.columns)}):")
        logger.info(f"\n{numeric.describe().round(2)}")

    # Categorical summary
    categorical = df.select_dtypes(include=["object", "category"])
    if len(categorical.columns) > 0:
        logger.info(f"\nCategorical features ({len(categorical.columns)}):")
        for col in categorical.columns:
            n_unique = categorical[col].nunique()
            logger.info(f"  {col}: {n_unique} unique values")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, cfg: dict) -> None:
    """Plot target class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    counts = df["target"].value_counts()
    colors = ["#22c55e", "#ef4444"]
    bars = ax.bar(
        ["No Default (0)", "Default (1)"],
        counts.values,
        color=colors,
        edgecolor="white",
    )

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{count}\n({count/len(df):.1%})",
            ha="center",
            fontsize=10,
        )

    ax.set_title("Target Distribution - Class Imbalance Check")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved target_distribution.png")


def plot_distributions(df: pd.DataFrame, cfg: dict) -> None:
    """Plot distribution of numeric features, colored by target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]

    if not numeric_cols:
        logger.warning("No numeric columns to plot distributions for")
        return

    n_cols = min(len(numeric_cols), 12)
    cols_to_plot = numeric_cols[:n_cols]
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        for target_val, color, label in [(0, "#22c55e", "No Default"), (1, "#ef4444", "Default")]:
            subset = df[df["target"] == target_val][col].dropna()
            ax.hist(subset, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=7)

    # Hide unused axes
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions by Target Class", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature_distributions.png")


def plot_correlation(df: pd.DataFrame, cfg: dict) -> None:
    """Plot correlation heatmap for numeric features."""
    numeric = df.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True if corr.shape[0] <= 15 else False,
        fmt=".2f" if corr.shape[0] <= 15 else "",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )

    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")


def plot_target_rate(df: pd.DataFrame, cfg: dict) -> None:
    """Plot default rate by categorical feature segments."""
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical:
        logger.info("No categorical columns for target rate analysis")
        return

    n_cols = min(len(categorical), 6)
    cols_to_plot = categorical[:n_cols]
    n_rows = (n_cols + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        rates = df.groupby(col)["target"].mean().sort_values(ascending=False)

        # Show top 10 categories if too many
        if len(rates) > 10:
            rates = rates.head(10)

        bars = ax.barh(range(len(rates)), rates.values, color="#3b82f6")
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels(rates.index, fontsize=8)
        ax.set_xlabel("Default Rate")
        ax.set_title(f"Default Rate by {col}", fontsize=10)
        ax.axvline(x=df["target"].mean(), color="red", linestyle="--", alpha=0.7, label="Overall")
        ax.legend(fontsize=7)

    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Default Rate by Categorical Segments", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_rate_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved target_rate_by_segment.png")