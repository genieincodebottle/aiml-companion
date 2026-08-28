"""PRIMARY MODEL: one LightGBM multiclass over time-to-payment buckets.

Answers both business questions from a single calibrated distribution:
  P(lapse)              -> proba[:, lapse_index]
  "when will it arrive"  -> the bucket distribution / expected_days()
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression

from lapse_prediction.config import CFG
from lapse_prediction.features.build import CATEGORICAL, assert_no_leakage, feature_columns


class BucketModel:
    def __init__(self, cfg=CFG, **params):
        self.cfg = cfg
        self.params = dict(
            objective="multiclass", num_class=cfg.n_classes,
            n_estimators=1500, learning_rate=0.04, num_leaves=63,
            min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1,
            random_state=cfg.seed, verbosity=-1, **params)
        self.model = None
        self.cols = None
        self._iso = None  # isotonic recalibration of the lapse probability

    # ------------------------------------------------------------------
    def fit(self, train, valid=None):
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        self.model = LGBMClassifier(**self.params)
        kw = {}
        if valid is not None and len(valid):
            kw = dict(eval_set=[(valid[self.cols], valid["bucket"])],
                      eval_metric="multi_logloss",
                      callbacks=[early_stopping(100, verbose=False),
                                 log_evaluation(0)])
        self.model.fit(train[self.cols], train["bucket"],
                       categorical_feature=CATEGORICAL, **kw)
        return self

    def predict_proba(self, df) -> np.ndarray:
        p = self.model.predict_proba(df[self.cols])
        if self._iso is not None:
            p = self._apply_iso(p)
        return p

    # ------------------------------------------------------------------
    def calibrate(self, holdout):
        """Fit isotonic on the lapse probability using a cohort the model never
        trained on, then rescale the remaining mass proportionally so the row
        still sums to 1. Business acts on these as risk scores, so the absolute
        level has to be right, not just the ranking."""
        raw = self.model.predict_proba(holdout[self.cols])
        self._iso = IsotonicRegression(out_of_bounds="clip").fit(
            raw[:, self.cfg.lapse_index], holdout["lapsed"].values)
        return self

    def _apply_iso(self, p):
        li = self.cfg.lapse_index
        new_lapse = np.clip(self._iso.predict(p[:, li]), 1e-6, 1 - 1e-6)
        rest = p[:, :li]
        rest = rest / np.clip(rest.sum(axis=1, keepdims=True), 1e-9, None)
        out = np.empty_like(p)
        out[:, :li] = rest * (1 - new_lapse)[:, None]
        out[:, li] = new_lapse
        return out

    # ------------------------------------------------------------------
    def importances(self, top: int = 20):
        import pandas as pd
        return (pd.Series(self.model.feature_importances_, index=self.cols)
                .sort_values(ascending=False).head(top))
