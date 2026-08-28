"""Fixing the probabilities without touching the ranking.

This is the resolution of the project's central tension. Naive Bayes on
dependent features is a good classifier and a terrible probability estimator.
Those are separable problems: a calibrator is a monotone map applied to the
scores, so it can move every probability while leaving the ORDER untouched.

Which is why the honest summary of Naive Bayes is not "it works anyway". It is
"its ranking survives the broken assumption, its probabilities do not, and you
can repair the half that broke".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from support_ticket_triage.config import CFG, Config
from support_ticket_triage.models.strategies import Strategy
from support_ticket_triage.utils.logging import get_logger

log = get_logger(__name__)


class CalibratedNB(Strategy):
    """MultinomialNB wrapped in cross-validated isotonic (or sigmoid) calibration.

    Note what this does NOT do: it does not make the independence assumption
    true, and it does not change which class wins on most rows. It maps the
    over-extreme scores back onto probabilities that mean what they say.
    """

    name = "calibrated_nb"

    def _fit(self, X, y):
        base = MultinomialNB(alpha=self.cfg.alpha)
        self.model = CalibratedClassifierCV(
            base, method=self.cfg.calibration_method, cv=self.cfg.calibration_cv,
        ).fit(X, y)
        self.classes_ = self.model.classes_
        self.n_submodels = self.cfg.calibration_cv

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def ranking_preserved(before: np.ndarray, after: np.ndarray,
                      classes: np.ndarray) -> dict:
    """How much did calibration change the actual decisions?

    The claim being tested is that calibration repairs probabilities without
    rewriting the classifier. If the top-1 agreement is high, the claim holds
    and the accuracy difference is noise rather than a different model.
    """
    top_before = classes[np.argmax(before, axis=1)]
    top_after = classes[np.argmax(after, axis=1)]
    agree = float((top_before == top_after).mean())
    return {
        "top1_agreement": round(agree, 4),
        "rows_changed": int((top_before != top_after).sum()),
        "mean_confidence_before": round(float(before.max(axis=1).mean()), 4),
        "mean_confidence_after": round(float(after.max(axis=1).mean()), 4),
    }


def confidence_shift(before: np.ndarray, after: np.ndarray,
                     bins: tuple[float, ...] = (0.5, 0.9, 0.99, 0.999, 1.01)
                     ) -> pd.DataFrame:
    """Where the confidence mass sat before and after.

    The interesting row is the top one. Uncalibrated Naive Bayes parks a large
    share of its predictions above 0.999, which is a claim of one error in a
    thousand that the accuracy column flatly contradicts.
    """
    cb, ca = before.max(axis=1), after.max(axis=1)
    rows, lo = [], 0.0
    for hi in bins:
        rows.append({
            "confidence_band": f"[{lo:.3f}, {hi:.3f})",
            "share_before": round(float(((cb >= lo) & (cb < hi)).mean()), 4),
            "share_after": round(float(((ca >= lo) & (ca < hi)).mean()), 4),
        })
        lo = hi
    return pd.DataFrame(rows)
