"""PRIMARY MODEL: one LightGBM multiclass over time-to-payment buckets.

Answers both business questions from a single calibrated distribution:
  P(lapse)              -> proba[:, lapse_index]
  "when will it arrive"  -> the bucket distribution / expected_days()
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from lapse_prediction.config import CFG
from lapse_prediction.features.build import CATEGORICAL, assert_no_leakage, feature_columns
from lapse_prediction.models.base import Base


class BucketModel(Base):
    """Calibration is inherited from `Base` rather than reimplemented here.

    It used to carry its own copy of the isotonic fit and the mass-rescaling
    arithmetic. Two copies of a subtle numerical routine is two places to fix
    when the routine turns out to be wrong -- and it was: see Base.calibrate
    for why this project no longer defaults to isotonic.
    """

    name = "bucket"

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

    def _raw_proba(self, df) -> np.ndarray:
        return self.model.predict_proba(df[self.cols])

    def predict_proba(self, df) -> np.ndarray:
        return self._apply_calibration(self._raw_proba(df))

    # ------------------------------------------------------------------
    def importances(self, top: int = 20):
        import pandas as pd
        return (pd.Series(self.model.feature_importances_, index=self.cols)
                .sort_values(ascending=False).head(top))
