"""
Data Loading & Cleaning
=======================

Loads credit risk data from multiple sources (OpenML German Credit,
local CSV, or Kaggle) and applies cleaning steps: type coercion,
missing value audit, column renaming, and target encoding.

Functions
---------
load_config     Load pipeline configuration from YAML.
load_data       Load raw data from configured source.
clean_data      Apply cleaning transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> dict:
    """Load pipeline configuration from YAML.

    Parameters
    ----------
    path : str or Path, optional
        Path to YAML config file. Defaults to ``configs/base.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "base.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against project root
    for key in ("raw_data_path", "output_dir"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(PROJECT_ROOT / cfg[key])

    logger.info(f"Loaded config from {path}")
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: dict) -> pd.DataFrame:
    """Load raw credit risk data from the configured source.

    Supports three sources (set ``data_source`` in config):
    - ``"openml"``  : German Credit dataset from OpenML (dataset 31)
    - ``"csv"``     : Local CSV file at ``raw_data_path``
    - ``"kaggle"``  : Kaggle dataset (requires kaggle CLI configured)

    Parameters
    ----------
    cfg : dict
        Pipeline configuration with ``data_source`` key.

    Returns
    -------
    pd.DataFrame
        Raw data with original column names.
    """
    source = cfg.get("data_source", "openml")
    logger.info(f"Loading data from source: {source}")

    if source == "openml":
        df = _load_openml(cfg)
    elif source == "csv":
        df = _load_csv(cfg)
    elif source == "kaggle":
        df = _load_kaggle(cfg)
    else:
        raise ValueError(f"Unknown data source: {source}")

    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def _load_openml(cfg: dict) -> pd.DataFrame:
    """Load German Credit dataset from OpenML (ID 31)."""
    from sklearn.datasets import fetch_openml

    dataset_id = cfg.get("openml_id", 31)
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
    df = data.frame.copy()

    # OpenML German Credit uses 1=good, 2=bad -> remap to 0/1
    if "class" in df.columns:
        df["target"] = (df["class"] == "bad").astype(int)
        df = df.drop(columns=["class"])

    return df


def _load_csv(cfg: dict) -> pd.DataFrame:
    """Load data from a local CSV file."""
    path = Path(cfg["raw_data_path"])
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _load_kaggle(cfg: dict) -> pd.DataFrame:
    """Load data from Kaggle (requires kaggle CLI)."""
    import subprocess
    import tempfile

    dataset = cfg.get("kaggle_dataset", "uciml/german-credit")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", tmp, "--unzip"],
            check=True,
            capture_output=True,
        )
        csv_files = list(Path(tmp).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found after downloading {dataset}")
        return pd.read_csv(csv_files[0])


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply cleaning transformations to raw data.

    Steps:
    1. Drop duplicate rows
    2. Coerce numeric columns
    3. Audit and log missing value patterns (MCAR vs MNAR)
    4. Rename columns to snake_case
    5. Validate target column exists

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    cfg : dict
        Pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Cleaned data ready for EDA.
    """
    df = df.copy()
    n_before = len(df)

    # 1. Drop exact duplicates
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.info(f"Dropped {n_dupes} duplicate rows")

    # 2. Coerce numeric columns (errors -> NaN)
    numeric_cols = cfg.get("numeric_columns", [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Missing value audit
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        logger.info("Missing values detected:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            pattern = _classify_missing(df, col)
            logger.info(f"  {col}: {count} ({pct:.1f}%) - {pattern}")

    # 4. Rename columns to snake_case
    df.columns = [_to_snake_case(c) for c in df.columns]

    # 5. Validate target
    target_col = cfg.get("target_column", "target")
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    logger.info(f"Cleaning complete: {len(df)} rows, {df.shape[1]} columns")
    return df


def _classify_missing(df: pd.DataFrame, col: str) -> str:
    """Classify missing value pattern as MCAR, MAR, or MNAR.

    Uses a simple heuristic: if missingness correlates with the target
    variable, it is likely MNAR. Otherwise MCAR.
    """
    if "target" not in df.columns:
        return "MCAR (no target to test)"

    missing_mask = df[col].isnull()
    if missing_mask.sum() == 0:
        return "No missing"

    # Compare target rate in missing vs non-missing groups
    rate_missing = df.loc[missing_mask, "target"].mean()
    rate_present = df.loc[~missing_mask, "target"].mean()

    if abs(rate_missing - rate_present) > 0.05:
        return "MNAR (target rate differs by >{:.0f}%)".format(
            abs(rate_missing - rate_present) * 100
        )
    return "MCAR (target rate similar)"


def _to_snake_case(name: str) -> str:
    """Convert column name to snake_case."""
    import re
    # Insert underscore before uppercase letters
    s = re.sub(r"([A-Z])", r"_\1", name)
    # Replace non-alphanumeric with underscore
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    # Remove leading/trailing underscores and collapse multiples
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()