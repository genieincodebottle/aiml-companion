"""Metrics that keep ranking and probability apart.

The whole project turns on one distinction that accuracy alone hides:

  * Is the model good at ORDERING the classes?  accuracy, macro-F1, rare recall
  * Is the model good at SAYING HOW SURE it is? Brier, ECE, mean top-1 confidence

Naive Bayes on dependent features scores well on the first and appallingly on
the second. If you only ever print accuracy you will never notice, and you will
then build a routing rule on a confidence number that means nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from support_ticket_triage.config import CFG, Config


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               classes: np.ndarray, n_bins: int = 12) -> float:
    """ECE on the top-1 prediction.

    Bin predictions by their confidence, then compare the average confidence in
    each bin against how often that bin was actually right. A perfectly
    calibrated model sits on the diagonal: when it says 0.9 it is right 90% of
    the time. ECE is the average gap, weighted by how many predictions land in
    each bin.
    """
    conf = proba.max(axis=1)
    pred = classes[np.argmax(proba, axis=1)]
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_table(y_true: np.ndarray, proba: np.ndarray,
                      classes: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    """The bin-by-bin numbers behind the ECE, for plotting or for reading."""
    conf = proba.max(axis=1)
    pred = classes[np.argmax(proba, axis=1)]
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        rows.append({
            "bin": f"({lo:.2f}, {hi:.2f}]",
            "n": int(m.sum()),
            "mean_confidence": round(float(conf[m].mean()), 4),
            "actual_accuracy": round(float(correct[m].mean()), 4),
            "gap": round(float(conf[m].mean() - correct[m].mean()), 4),
        })
    return pd.DataFrame(rows)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray,
                     classes: np.ndarray) -> float:
    """Mean squared error between the probability vector and a one-hot truth.

    Unlike accuracy this is a PROPER scoring rule: it is minimised only by
    telling the truth about your uncertainty, so overconfidence is punished
    even when the argmax is right.
    """
    onehot = np.zeros_like(proba)
    index = {c: i for i, c in enumerate(classes)}
    for r, y in enumerate(y_true):
        onehot[r, index[y]] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def score(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
          name: str = "", rare_class: str = "abuse_report",
          cfg: Config = CFG) -> dict:
    """Every headline number for one model, ranking and probability side by side."""
    pred = classes[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    rare_recall = (recall_score(y_true, pred, labels=[rare_class],
                                average="macro", zero_division=0)
                   if rare_class in set(classes) else float("nan"))
    return {
        "model": name,
        # ranking quality
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro",
                                         zero_division=0)), 4),
        f"recall_{rare_class}": round(float(rare_recall), 4),
        # probability quality
        "brier": round(multiclass_brier(y_true, proba, classes), 4),
        "ece": round(expected_calibration_error(
            y_true, proba, classes, cfg.n_calibration_bins), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "pct_over_99pct_sure": round(float((conf > 0.99).mean()), 4),
    }


def per_class_table(y_true: np.ndarray, proba: np.ndarray,
                    classes: np.ndarray) -> pd.DataFrame:
    """Recall and support per class, so the 3% class cannot hide inside accuracy."""
    pred = classes[np.argmax(proba, axis=1)]
    rows = []
    for c in classes:
        m = y_true == c
        rows.append({
            "class": c,
            "support": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "recall": round(float((pred[m] == c).mean()) if m.any() else 0.0, 4),
            "precision": round(
                float((y_true[pred == c] == c).mean()) if (pred == c).any() else 0.0, 4),
        })
    return pd.DataFrame(rows).sort_values("share", ascending=False,
                                          ignore_index=True)
