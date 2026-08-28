"""The interface every model in this package implements.

    m.fit(train, valid=None) -> self
    m.predict_proba(df)      -> (n_rows, n_classes) over CFG.class_names

Multiclass, ordinal chain, hurdle, AFT, Cox, discrete hazard and neural models
all reduce to the same bucket distribution, which is what makes the bake-off
in pipelines/benchmark.py an apples-to-apples comparison and what lets serving
code swap one for another without changing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG


class Base:
    """Subclasses set `name` and implement fit/predict_proba.

    Calibration lives here rather than on individual models: ops reads
    `p_lapse` as a probability, so the LEVEL has to be right for whichever
    model is deployed, not only for the ones that happened to implement it.
    """
    name: str = "base"
    _iso = None

    # ------------------------------------------------------------ calibration
    def calibrate(self, holdout: pd.DataFrame) -> "Base":
        """Fit isotonic on the lapse probability using a cohort the model never
        trained on, then rescale the remaining bucket mass so rows still sum
        to 1. Must be given a holdout, never the training rows."""
        from sklearn.isotonic import IsotonicRegression
        raw = self._raw_proba(holdout)
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw[:, CFG.lapse_index], holdout["lapsed"].to_numpy())
        return self

    def _raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probabilities. Subclasses implement THIS; predict_proba
        applies calibration on top."""
        raise NotImplementedError

    def _apply_calibration(self, p: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return p
        li = CFG.lapse_index
        lapse = np.clip(self._iso.predict(p[:, li]), 1e-6, 1 - 1e-6)
        rest = p[:, :li]
        rest = rest / np.clip(rest.sum(axis=1, keepdims=True), 1e-9, None)
        out = np.empty_like(p)
        out[:, :li] = rest * (1 - lapse)[:, None]
        out[:, li] = lapse
        return out

    @property
    def is_calibrated(self) -> bool:
        return self._iso is not None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "Base":
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def from_cdf(F: np.ndarray, cfg=CFG) -> np.ndarray:
    """CDF at the bucket edges -> bucket probabilities plus the lapse tail.

    Monotonicity is enforced here rather than assumed: AFT, Cox and the ordinal
    chain can each emit a non-monotone curve, and a model that claims
    P(paid by day 15) < P(paid by day 7) cannot be shown to an ops user.
    """
    li = cfg.lapse_index
    F = np.clip(np.maximum.accumulate(np.clip(F, 0, 1), axis=1), 1e-6, 1 - 1e-6)
    p = np.empty((F.shape[0], cfg.n_classes))
    p[:, 0] = F[:, 0]
    p[:, 1:li] = np.diff(F, axis=1)
    p[:, li] = 1 - F[:, -1]
    return p / p.sum(axis=1, keepdims=True)
