"""Evaluation that matches how the business will actually use the scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)

from lapse_prediction.config import CFG


def lapse_metrics(y_lapse, p_lapse) -> dict:
    return {
        "lapse_rate": float(np.mean(y_lapse)),
        "roc_auc": float(roc_auc_score(y_lapse, p_lapse)),
        "pr_auc": float(average_precision_score(y_lapse, p_lapse)),
        "brier": float(brier_score_loss(y_lapse, p_lapse)),
        "log_loss": float(log_loss(y_lapse, np.clip(p_lapse, 1e-6, 1 - 1e-6))),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), bins, duplicates="drop", labels=False)
    return (pd.DataFrame({"bin": q, "p": p, "y": y})
            .groupby("bin")
            .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
            .assign(gap=lambda d: d.predicted - d.actual)
            .round(4))


def lift_table(y_lapse, p_lapse, deciles: int = 10) -> pd.DataFrame:
    """What a retention-calling queue actually cares about: capture-by-decile."""
    d = pd.DataFrame({"y": np.asarray(y_lapse), "p": np.asarray(p_lapse)})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False),
                          deciles, labels=False) + 1
    t = d.groupby("decile").agg(n=("y", "size"), lapses=("y", "sum"),
                                mean_p=("p", "mean"))
    t["lapse_rate"] = t["lapses"] / t["n"]
    t["cum_capture"] = t["lapses"].cumsum() / t["lapses"].sum()
    t["lift"] = t["lapse_rate"] / d["y"].mean()
    return t.round(4)


def expected_days(proba, cfg=CFG) -> np.ndarray:
    """Mean days-to-payment conditional on paying within grace.

    Uses bucket midpoints, renormalised over the non-lapse buckets. This is the
    number to show an ops user -- never a raw regression on skewed day counts.
    """
    mids, lo = [], 0
    for _, ub in cfg.buckets:
        mids.append((lo + ub) / 2.0)
        lo = ub + 1
    mids = np.asarray(mids)
    pay = proba[:, :cfg.lapse_index]
    denom = pay.sum(axis=1, keepdims=True)
    return (pay / np.clip(denom, 1e-9, None) @ mids)


def bucket_report(y_bucket, proba, cfg=CFG) -> pd.DataFrame:
    """Per-bucket predicted mass vs actual -- checks the whole distribution,
    not just the lapse tail."""
    y = np.asarray(y_bucket)
    return pd.DataFrame({
        "bucket": cfg.class_names,
        "actual_share": [float(np.mean(y == i)) for i in range(cfg.n_classes)],
        "predicted_share": proba.mean(axis=0),
        "auc_ovr": [float(roc_auc_score((y == i).astype(int), proba[:, i]))
                    if 0 < (y == i).mean() < 1 else np.nan
                    for i in range(cfg.n_classes)],
    }).round(4)


def days_mae(y_days, pred_days, mask) -> float:
    """MAE on days, among dues that were actually paid within grace."""
    m = np.asarray(mask, bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_days)[m] - np.asarray(pred_days)[m])))
