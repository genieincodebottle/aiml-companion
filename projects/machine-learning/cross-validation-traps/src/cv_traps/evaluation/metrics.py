"""Scoring, and the one derived number the whole project turns on.

`optimism` is a cross-validation estimate minus the truth it was estimating.
Positive means the scheme flattered the model. That single subtraction is what
turns "grouped CV is more correct" from an opinion into a measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC AUC, or NaN when a fold happens to be single-class.

    Returning NaN rather than raising matters: forward-chaining folds are
    period-sized and a rare-event panel can hand you one with no positives.
    Killing the run over that would make the honest scheme look fragile for a
    reason that has nothing to do with its honesty.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def cv_score(estimator, X: pd.DataFrame, y: np.ndarray,
             splits) -> tuple[float, list[float]]:
    """Mean AUC across folds, plus the per-fold values.

    The spread is returned because the mean alone is what lets people believe
    a 0.004 difference between two models is real.
    """
    scores = []
    for train_idx, test_idx in splits:
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y[train_idx])
        proba = est.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(auc(y[test_idx], proba))
    clean = [s for s in scores if not np.isnan(s)]
    return (float(np.mean(clean)) if clean else float("nan")), scores


def holdout_score(estimator, X_dev: pd.DataFrame, y_dev: np.ndarray,
                  X_out: pd.DataFrame, y_out: np.ndarray) -> float:
    """Fit on everything available, score on the truth. No folds involved."""
    est = clone(estimator)
    est.fit(X_dev, y_dev)
    return auc(y_out, est.predict_proba(X_out)[:, 1])


def optimism_table(rows: list[dict], truth: float) -> pd.DataFrame:
    """Attach the truth and the signed error to every estimate."""
    out = pd.DataFrame(rows)
    out["truth"] = round(truth, 4)
    out["optimism"] = (out["cv_auc"] - truth).round(4)
    out["cv_auc"] = out["cv_auc"].round(4)
    return out
