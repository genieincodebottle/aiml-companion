"""
data_loader.py - IPL Dataset Loading and Cleaning Utilities
============================================================

Provides functions to load IPL match-level and ball-by-ball datasets
from multiple sources (local files, Kaggle paths, or GitHub raw URLs)
and perform standard cleaning operations such as missing-value
imputation, type casting, and team-name standardization.

Usage
-----
>>> from src.data_loader import load_matches, load_deliveries, clean_team_names
>>> matches_df = load_matches()
>>> deliveries_df = load_deliveries()
>>> matches_df, deliveries_df = clean_team_names(matches_df, deliveries_df)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default constants (overridable via configs/base.yaml)
# ---------------------------------------------------------------------------
_MATCHES_URL = (
    "https://raw.githubusercontent.com/genieincodebottle/generative-ai/"
    "main/docs/ipl-dataset/matches.csv"
)
_DELIVERIES_URL = (
    "https://raw.githubusercontent.com/genieincodebottle/generative-ai/"
    "main/docs/ipl-dataset/deliveries.csv"
)
_KAGGLE_PATH = "/kaggle/input/ipl-complete-dataset-20082020"
_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
_NA_VALUES: List[str] = ["NA", ""]

_DEFAULT_TEAM_NAME_MAPPING: Dict[str, str] = {
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """Return the project root directory (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Absolute or project-relative path to the YAML config.
        Defaults to ``configs/base.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = str(_project_root() / "configs" / "base.yaml")
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found; using defaults.", config_path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Data-source resolution
# ---------------------------------------------------------------------------
def _resolve_source(
    filename: str,
    url: str,
    local_paths: Optional[List[str]] = None,
) -> str:
    """Resolve the best available data source for *filename*.

    Checks local paths first (faster), then falls back to the remote URL.

    Parameters
    ----------
    filename : str
        CSV file name, e.g. ``"matches.csv"``.
    url : str
        Remote URL to fall back on if no local file is found.
    local_paths : list of str, optional
        Directories to search for *filename* before using *url*.

    Returns
    -------
    str
        Path or URL string suitable for ``pd.read_csv``.
    """
    cache_dir = str(_project_root() / "data" / "raw")
    search_dirs = local_paths or [_KAGGLE_PATH, cache_dir]
    for directory in search_dirs:
        candidate = os.path.join(directory, filename)
        if os.path.exists(candidate):
            logger.info("Loading %s from local path: %s", filename, candidate)
            return candidate

    # No local file found - download from URL and cache locally
    logger.info(
        "No local copy of %s found. Downloading from GitHub (~%s)...",
        filename,
        "27 MB" if "deliveries" in filename else "224 KB",
    )
    print(
        f"Downloading {filename} from GitHub"
        f" (~{'27 MB' if 'deliveries' in filename else '224 KB'})..."
    )
    try:
        df = pd.read_csv(url, na_values=_NA_VALUES)
        os.makedirs(cache_dir, exist_ok=True)
        cached_path = os.path.join(cache_dir, filename)
        df.to_csv(cached_path, index=False)
        logger.info("Cached %s to %s for faster future runs.", filename, cached_path)
        print(f"Cached {filename} locally for faster future runs.")
        return cached_path
    except Exception as exc:
        logger.warning("Failed to cache %s locally: %s. Using URL directly.", filename, exc)
        return url


# ---------------------------------------------------------------------------
# Public loading functions
# ---------------------------------------------------------------------------
def load_matches(
    source: Optional[str] = None,
    na_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the IPL matches dataset.

    Parameters
    ----------
    source : str, optional
        Explicit file path or URL.  When *None*, the function auto-resolves
        the source by checking local paths and falling back to the GitHub
        raw URL.
    na_values : list of str, optional
        Additional strings to recognise as NA/NaN.  Defaults to
        ``["NA", ""]``.

    Returns
    -------
    pd.DataFrame
        Raw matches DataFrame (1 169 rows x 20 columns for the 2008-2025
        dataset).

    Examples
    --------
    >>> matches = load_matches()
    >>> matches.shape
    (1169, 20)
    """
    cfg = load_config()
    if source is None:
        url = cfg.get("data", {}).get("matches_url", _MATCHES_URL)
        source = _resolve_source("matches.csv", url)
    na_vals = na_values if na_values is not None else cfg.get("data", {}).get("na_values", _NA_VALUES)

    df = pd.read_csv(source, na_values=na_vals)
    logger.info(
        "Loaded matches dataset — shape: %s, columns: %s",
        df.shape,
        list(df.columns),
    )
    return df


def load_deliveries(
    source: Optional[str] = None,
    na_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the IPL deliveries (ball-by-ball) dataset.

    Parameters
    ----------
    source : str, optional
        Explicit file path or URL.  When *None*, the function auto-resolves
        the source by checking local paths and falling back to the GitHub
        raw URL.
    na_values : list of str, optional
        Additional strings to recognise as NA/NaN.  Defaults to
        ``["NA", ""]``.

    Returns
    -------
    pd.DataFrame
        Raw deliveries DataFrame (260 920 rows x 17 columns for the
        2008-2025 dataset).

    Examples
    --------
    >>> deliveries = load_deliveries()
    >>> deliveries.shape
    (260920, 17)
    """
    cfg = load_config()
    if source is None:
        url = cfg.get("data", {}).get("deliveries_url", _DELIVERIES_URL)
        source = _resolve_source("deliveries.csv", url)
    na_vals = na_values if na_values is not None else cfg.get("data", {}).get("na_values", _NA_VALUES)

    df = pd.read_csv(source, na_values=na_vals)
    logger.info(
        "Loaded deliveries dataset — shape: %s, columns: %s",
        df.shape,
        list(df.columns),
    )
    return df


def load_all(
    matches_source: Optional[str] = None,
    deliveries_source: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper to load both datasets at once.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        ``(matches_df, deliveries_df)``
    """
    return load_matches(matches_source), load_deliveries(deliveries_source)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def clean_team_names(
    matches_df: pd.DataFrame,
    deliveries_df: Optional[pd.DataFrame] = None,
    mapping: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Standardise team names across matches and deliveries DataFrames.

    Franchise renames (e.g. *Royal Challengers Bengaluru* -> *Royal
    Challengers Bangalore*) and spelling inconsistencies (e.g. *Rising
    Pune Supergiants* vs *Rising Pune Supergiant*) are normalised to a
    single canonical name.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Matches DataFrame with columns ``team1``, ``team2``,
        ``toss_winner``, ``winner``.
    deliveries_df : pd.DataFrame, optional
        Deliveries DataFrame with columns ``batting_team``,
        ``bowling_team``.  Pass *None* to skip.
    mapping : dict, optional
        Custom ``{old_name: new_name}`` mapping.  Defaults to the
        built-in mapping covering known renames up to 2025.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame or None)
        Cleaned ``(matches_df, deliveries_df)``.
    """
    name_map = mapping if mapping is not None else _DEFAULT_TEAM_NAME_MAPPING

    match_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in match_cols:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].replace(name_map)

    if deliveries_df is not None:
        delivery_cols = ["batting_team", "bowling_team"]
        for col in delivery_cols:
            if col in deliveries_df.columns:
                deliveries_df[col] = deliveries_df[col].replace(name_map)

    logger.info("Standardised team names using mapping: %s", name_map)
    return matches_df, deliveries_df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps to the matches DataFrame.

    Cleaning steps performed:
    1. Fill missing ``winner`` with ``'No Result'`` where
       ``result == 'no result'``, otherwise ``'Unknown'``.
    2. Fill missing ``player_of_match`` with ``'Not Awarded'``.
    3. Fill missing ``city`` with ``'Unknown'``.
    4. Fill missing ``result_margin`` with ``0``.
    5. Fill missing ``method`` with ``'Normal'``.
    6. Fill missing umpire columns with ``'Unknown'``.
    7. Derive ``win_by_runs`` and ``win_by_wickets`` from ``result``
       and ``result_margin``.
    8. Convert ``date`` to ``datetime``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw matches DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned matches DataFrame with additional derived columns.
    """
    cleaned = df.copy()

    # --- Missing value imputation ---
    no_result_mask = cleaned["result"] == "no result"
    cleaned.loc[no_result_mask, "winner"] = cleaned.loc[
        no_result_mask, "winner"
    ].fillna("No Result")
    cleaned.loc[no_result_mask, "player_of_match"] = cleaned.loc[
        no_result_mask, "player_of_match"
    ].fillna("Not Awarded")

    cleaned["winner"] = cleaned["winner"].fillna("Unknown")
    cleaned["player_of_match"] = cleaned["player_of_match"].fillna("Not Awarded")
    cleaned["city"] = cleaned["city"].fillna("Unknown")
    cleaned["result_margin"] = cleaned["result_margin"].fillna(0)
    cleaned["method"] = cleaned["method"].fillna("Normal")

    for ump_col in ("umpire1", "umpire2"):
        if ump_col in cleaned.columns:
            cleaned[ump_col] = cleaned[ump_col].fillna("Unknown")

    # --- Derived columns ---
    cleaned["win_by_runs"] = np.where(
        cleaned["result"] == "runs", cleaned["result_margin"], 0
    )
    cleaned["win_by_wickets"] = np.where(
        cleaned["result"] == "wickets", cleaned["result_margin"], 0
    )

    # --- Type conversions ---
    cleaned["date"] = pd.to_datetime(cleaned["date"], format="%Y-%m-%d", errors="coerce")

    # --- Convert binary string columns to int ---
    if "super_over" in cleaned.columns:
        cleaned["super_over"] = cleaned["super_over"].map({"Y": 1, "N": 0}).fillna(0).astype(int)
    if "dl_applied" in cleaned.columns:
        cleaned["dl_applied"] = cleaned["dl_applied"].astype(int)

    # --- Toss-winner-won-match flag ---
    cleaned["toss_winner_won_match"] = (
        cleaned["toss_winner"] == cleaned["winner"]
    ).astype(int)

    logger.info(
        "Cleaned matches DataFrame — remaining nulls:\n%s",
        cleaned.isnull().sum().to_string(),
    )
    return cleaned


def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps to the deliveries DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw deliveries DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned deliveries DataFrame.
    """
    cleaned = df.copy()

    # Total runs per ball (batsman + extras)
    if "total_runs" not in cleaned.columns and "batsman_runs" in cleaned.columns:
        extras = cleaned.get("extra_runs", 0)
        cleaned["total_runs"] = cleaned["batsman_runs"] + extras

    # Boolean flags for boundaries
    if "batsman_runs" in cleaned.columns:
        cleaned["is_four"] = (cleaned["batsman_runs"] == 4).astype(int)
        cleaned["is_six"] = (cleaned["batsman_runs"] == 6).astype(int)

    logger.info(
        "Cleaned deliveries DataFrame — shape: %s", cleaned.shape
    )
    return cleaned


def get_unique_teams(matches_df: pd.DataFrame) -> List[str]:
    """Return a sorted list of unique team names from the matches data.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Matches DataFrame (cleaned or raw).

    Returns
    -------
    list of str
        Sorted unique team names.
    """
    teams = set()
    for col in ("team1", "team2"):
        if col in matches_df.columns:
            teams.update(matches_df[col].dropna().unique())
    return sorted(teams)
