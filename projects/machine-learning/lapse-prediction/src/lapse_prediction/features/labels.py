"""Targets + cohort maturity + out-of-time splitting."""
from __future__ import annotations

import pandas as pd

from lapse_prediction.config import CFG


def add_labels(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = df["days_to_pay"].map(cfg.bucket_of).astype(int)
    df["lapsed"] = (df["bucket"] == cfg.lapse_index).astype(int)
    return df


def mature(df: pd.DataFrame, as_of=None, cfg=CFG) -> pd.DataFrame:
    """Keep only cohorts whose grace period has fully elapsed.

    Training on immature cohorts silently relabels not-yet-paid as lapsed and
    is the single most common way this model gets quietly ruined.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else df["due_date"].max()
    return df[df["due_date"] + pd.Timedelta(days=cfg.grace_days) <= as_of].copy()


def time_split(df: pd.DataFrame, cfg=CFG):
    """Split by due-date cohort: train | test (out-of-time) | valid (most recent)."""
    end = df["due_date"].max()
    v0 = end - pd.DateOffset(months=cfg.valid_months)
    t0 = v0 - pd.DateOffset(months=cfg.test_months)
    return (df[df["due_date"] < t0].copy(),
            df[(df["due_date"] >= t0) & (df["due_date"] < v0)].copy(),
            df[df["due_date"] >= v0].copy())
