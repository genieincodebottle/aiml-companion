"""Evaluation report: one function that produces every number worth logging,
so training, benchmarking and monitoring all report the same things."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from lapse_prediction.config import CFG, Config
from lapse_prediction.evaluation import metrics


def monotonicity_violation(proba: np.ndarray, cfg: Config = CFG) -> float:
    F = np.cumsum(proba[:, :cfg.lapse_index], axis=1)
    return float((np.diff(F, axis=1) < -1e-9).any(axis=1).mean())


def full_report(valid: pd.DataFrame, proba: np.ndarray, cfg: Config = CFG) -> dict:
    y_l = valid["lapsed"].to_numpy()
    li = cfg.lapse_index
    eta = metrics.expected_days(proba, cfg)
    paid = y_l == 0
    lift = metrics.lift_table(y_l, proba[:, li])
    head = metrics.lapse_metrics(y_l, proba[:, li])

    headline = {
        "lapse_pr_auc": round(head["pr_auc"], 4),
        "lapse_roc_auc": round(head["roc_auc"], 4),
        "lapse_brier": round(head["brier"], 4),
        "bucket_mlogloss": round(float(log_loss(
            valid["bucket"], np.clip(proba, 1e-9, 1),
            labels=list(range(cfg.n_classes)))), 4),
        "days_mae": round(metrics.days_mae(valid["days_to_pay"], eta, paid), 2),
        "capture_at_20pct": round(float(lift["cum_capture"].iloc[1]), 4),
        "monotonicity_violation": round(monotonicity_violation(proba, cfg), 4),
        "n_valid": int(len(valid)),
        "lapse_rate": round(float(y_l.mean()), 4),
    }
    return {
        "headline": headline,
        "calibration": metrics.calibration_table(y_l, proba[:, li]),
        "lift": lift,
        "buckets": metrics.bucket_report(valid["bucket"], proba, cfg),
    }


def log_report(scores: dict, log) -> None:
    log.info("headline: %s", scores["headline"])
    for k in ("buckets", "calibration", "lift"):
        log.info("%s\n%s", k, scores[k].to_string())


def gate(headline: dict, min_pr_auc: float = 0.25, max_brier: float = 0.12,
         max_mono_violation: float = 0.0) -> tuple[bool, list[str]]:
    """Release gate. A pipeline that trains a bad model and ships it anyway is
    worse than one that fails loudly, so make the bar explicit."""
    fails = []
    if headline["lapse_pr_auc"] < min_pr_auc:
        fails.append(f"PR-AUC {headline['lapse_pr_auc']} < {min_pr_auc}")
    if headline["lapse_brier"] > max_brier:
        fails.append(f"Brier {headline['lapse_brier']} > {max_brier}")
    if headline["monotonicity_violation"] > max_mono_violation:
        fails.append(f"monotonicity violated on "
                     f"{headline['monotonicity_violation']:.1%} of rows")
    return (not fails), fails
