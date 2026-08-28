"""BASELINE: the two-stage hurdle the original question proposed.

Stage 1: lapse / no-lapse binary.
Stage 2: trained ONLY on non-lapsed dues -> which time bucket.
Chained at inference: P(bucket_j) = (1 - P(lapse)) * P(bucket_j | paid).

Kept so the choice is measured, not asserted. Its known weakness is that
stage 2 never sees stage-1 errors, and the two stages can be miscalibrated
against each other; the single multiclass model gets one joint distribution.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier

from lapse_prediction.config import CFG
from lapse_prediction.features.build import CATEGORICAL, assert_no_leakage, feature_columns


class HurdleModel:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        base = dict(n_estimators=800, learning_rate=0.05, num_leaves=63,
                    min_child_samples=80, colsample_bytree=0.8, reg_lambda=5.0,
                    n_jobs=-1, random_state=cfg.seed, verbosity=-1)
        self.m1 = LGBMClassifier(objective="binary", **base)
        self.m2 = LGBMClassifier(objective="multiclass",
                                 num_class=cfg.lapse_index, **base)
        self.cols = None

    def fit(self, train):
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        self.m1.fit(train[self.cols], train["lapsed"],
                    categorical_feature=CATEGORICAL)
        paid = train[train["lapsed"] == 0]
        self.m2.fit(paid[self.cols], paid["bucket"],
                    categorical_feature=CATEGORICAL)
        return self

    def predict_proba(self, df) -> np.ndarray:
        p_lapse = self.m1.predict_proba(df[self.cols])[:, 1]
        p_bucket = self.m2.predict_proba(df[self.cols])
        out = np.empty((len(df), self.cfg.n_classes))
        out[:, :self.cfg.lapse_index] = p_bucket * (1 - p_lapse)[:, None]
        out[:, self.cfg.lapse_index] = p_lapse
        return out
