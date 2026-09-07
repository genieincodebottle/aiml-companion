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


def split_oot_cohort(test: pd.DataFrame, cfg=CFG):
    """Divide the out-of-time cohort into (early_stop, calibration), by date.

    Two different jobs need held-out rows during training, and they need
    DIFFERENT held-out rows:

      * early stopping picks the number of boosting rounds by watching a
        cohort, which makes that cohort partly in-sample
      * the calibrator maps raw scores onto honest probabilities, and has to
        see scores as fresh data will see them

    Reuse one cohort for both and the calibrator is fitted on scores the model
    was tuned to get right, so it corrects a distortion that does not exist off
    that cohort. Split by due date rather than at random, for the same reason
    the outer split is temporal: the calibration cohort should sit after the
    early-stopping one, the way deployment sits after training.
    """
    if test.empty:
        return test, test
    ordered = test.sort_values("due_date")
    cut = int(len(ordered) * cfg.early_stop_share)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
