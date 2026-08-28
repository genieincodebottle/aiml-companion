"""SECONDARY MODEL: discrete-time survival hazard, same boosted-tree tooling.

Use this when the horizon runs past the grace period i.e. when you care about
revivals (payments arriving 3, 6, 12 months late). It handles right-censoring
properly, so recent cohorts do NOT have to be thrown away: a due date 20 days
old contributes its first 20 days of exposure and is censored after that.

Mechanics: expand each due into one row per period, label = "paid in this
period", fit a binary classifier on the hazard h(t|x), then
    S(t) = prod_{k<=t} (1 - h(k))      P(pay by t) = 1 - S(t)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from lapse_prediction.config import CFG
from lapse_prediction.features.build import CATEGORICAL, feature_columns


def expand(df: pd.DataFrame, as_of=None, cfg=CFG, sample: float = 1.0,
           seed: int = CFG.seed) -> pd.DataFrame:
    """Person-period table with both event and administrative censoring."""
    d = df if sample >= 1.0 else df.sample(frac=sample, random_state=seed)
    as_of = pd.Timestamp(as_of) if as_of is not None else d["due_date"].max()
    P, K = cfg.hazard_period_days, cfg.hazard_horizon_days // cfg.hazard_period_days

    days = d["days_to_pay"].to_numpy(dtype=float)
    event_period = np.where(np.isnan(days) | (days > cfg.hazard_horizon_days),
                            np.inf, np.floor(days / P))
    # administrative censoring: how many periods have actually been observed
    observed = np.minimum(
        K, np.floor((as_of - d["due_date"]).dt.days.to_numpy() / P))
    last = np.minimum(np.where(np.isinf(event_period), K - 1, event_period),
                      observed - 1)

    keep = last >= 0
    d, last, event_period = d[keep], last[keep], event_period[keep]
    n_periods = (last + 1).astype(int)

    idx = np.repeat(np.arange(len(d)), n_periods)
    period = np.concatenate([np.arange(n) for n in n_periods])
    out = d.iloc[idx].reset_index(drop=True)
    out["period"] = period
    out["paid_this_period"] = (period == np.repeat(
        np.where(np.isinf(event_period), -1, event_period), n_periods)).astype(int)
    out["days_elapsed"] = period * P
    out["past_grace"] = (out["days_elapsed"] > cfg.grace_days).astype(int)
    return out


class HazardModel:
    def __init__(self, cfg=CFG, **params):
        self.cfg = cfg
        defaults = dict(objective="binary", n_estimators=800,
                        learning_rate=0.05, num_leaves=63,
                        min_child_samples=200, colsample_bytree=0.8,
                        reg_lambda=5.0, n_jobs=-1, random_state=cfg.seed,
                        verbosity=-1)
        self.params = {**defaults, **params}
        self.model, self.cols = None, None

    def fit(self, pp: pd.DataFrame):
        self.cols = [c for c in feature_columns(pp) if c != "paid_this_period"]
        self.model = LGBMClassifier(**self.params).fit(
            pp[self.cols], pp["paid_this_period"],
            categorical_feature=CATEGORICAL)
        return self

    def survival(self, df: pd.DataFrame, horizon_days: int | None = None):
        """Returns (S, days) where S[i, t] = P(still unpaid after period t)."""
        cfg = self.cfg
        P = cfg.hazard_period_days
        K = (horizon_days or cfg.hazard_horizon_days) // P
        h = np.empty((len(df), K))
        base = df.reset_index(drop=True)
        for k in range(K):
            x = base.copy()
            x["period"] = k
            x["days_elapsed"] = k * P
            x["past_grace"] = int(k * P > cfg.grace_days)
            h[:, k] = self.model.predict_proba(x[self.cols])[:, 1]
        return np.cumprod(1 - h, axis=1), (np.arange(K) + 1) * P

    def p_paid_by(self, df, day: int, survival=None) -> np.ndarray:
        """P(premium received on or before `day`). Pass a precomputed
        `survival` tuple to avoid re-running the curve for each horizon."""
        S, days = survival if survival is not None else self.survival(df)
        idx = int(np.clip(np.searchsorted(days, day, side="right") - 1,
                          0, S.shape[1] - 1))
        return 1 - S[:, idx]

    def median_days(self, df, survival=None) -> np.ndarray:
        """Median time-to-payment; NaN when >50% never pay in the horizon."""
        S, days = survival if survival is not None else self.survival(df)
        hit = S <= 0.5
        out = np.where(hit.any(axis=1), days[hit.argmax(axis=1)], np.nan)
        return out
