"""As-of-due-date feature engineering.

LEAKAGE RULE, enforced structurally: every history feature is built from rows
*strictly before* the current due date (groupby.shift(1) then expanding/rolling).
Nothing about the current due event -- least of all `days_to_pay` -- may appear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG

CATEGORICAL = ["premium_freq", "payment_mode", "product", "channel"]
_LEAKY = {"days_to_pay", "paid_date", "bucket", "lapsed", "payment_date"}


def build(df: pd.DataFrame, cfg=CFG) -> pd.DataFrame:
    df = df.sort_values(["policy_id", "due_date"]).copy()
    g = df.groupby("policy_id", sort=False)

    # grace-capped days-late of the *previous* dues -> the strongest signal
    capped = df["days_to_pay"].clip(upper=cfg.grace_days).fillna(cfg.grace_days)
    was_lapse = (df["days_to_pay"].isna() | (df["days_to_pay"] > cfg.grace_days)).astype(int)
    df["_capped"], df["_lapse"] = capped, was_lapse

    prev = g["_capped"].shift(1)
    for k in (1, 2, 3):
        df[f"days_late_lag{k}"] = g["_capped"].shift(k)
    df["days_late_mean3"] = g["_capped"].shift(1).groupby(df["policy_id"]).rolling(
        3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["days_late_max3"] = g["_capped"].shift(1).groupby(df["policy_id"]).rolling(
        3, min_periods=1).max().reset_index(level=0, drop=True)
    df["days_late_std3"] = g["_capped"].shift(1).groupby(df["policy_id"]).rolling(
        3, min_periods=2).std().reset_index(level=0, drop=True)
    df["days_late_trend"] = prev - df["days_late_lag2"]

    df["prior_dues"] = g.cumcount()
    df["prior_lapses"] = g["_lapse"].shift(1).groupby(df["policy_id"]).cumsum().fillna(0)
    df["prior_lapse_rate"] = df["prior_lapses"] / df["prior_dues"].replace(0, np.nan)
    df["ontime_rate"] = (g["_capped"].shift(1).le(7).groupby(df["policy_id"])
                         .expanding().mean().reset_index(level=0, drop=True))
    df["ever_lapsed"] = (df["prior_lapses"] > 0).astype(int)
    df["days_since_prev_due"] = (df["due_date"] - g["due_date"].shift(1)).dt.days

    # exposure / seasonality / money
    df["tenure_days"] = df["policy_year"] * 365
    df["due_month"] = df["due_date"].dt.month
    df["due_quarter"] = df["due_date"].dt.quarter
    df["is_first_renewal"] = (df["prior_dues"] == 0).astype(int)
    df["log_premium"] = np.log1p(df["annual_premium"])
    df["premium_per_year_of_age"] = df["annual_premium"] / df["cust_age"]
    df["sum_assured"] = df["annual_premium"] * df["sum_assured_mult"]
    df["agent_active"] = df["agent_active"].astype(int)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    return df.drop(columns=["_capped", "_lapse"])


def feature_columns(df: pd.DataFrame) -> list:
    drop = _LEAKY | {"policy_id", "due_date"}
    return [c for c in df.columns if c not in drop]


def assert_no_leakage(cols) -> None:
    bad = _LEAKY.intersection(cols)
    if bad:
        raise ValueError(f"leaky features in matrix: {sorted(bad)}")
