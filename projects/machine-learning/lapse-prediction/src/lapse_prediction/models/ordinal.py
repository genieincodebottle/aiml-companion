"""The ordinal cumulative chain -- the recommended in-grace model.

The buckets are ORDERED and multiclass softmax throws that ordering away. Here
one binary model per bucket edge answers a question the business asks in plain
words -- "will the premium be in by day 7? by 15? by 30? by grace-end?" --
and differencing the monotonised cumulative curve reconstructs the bucket
distribution. Every sub-model trains on all rows (unlike the hurdle, whose
second stage sees only non-lapsers), and P(lapse) = 1 - F(grace) is a
chain-consistent tail rather than a separately fitted head that can disagree.
"""
from __future__ import annotations

import numpy as np

from lapse_prediction.config import CFG
from lapse_prediction.features.build import (CATEGORICAL, assert_no_leakage,
                                             feature_columns)
from lapse_prediction.models.base import Base, from_cdf

EDGES = np.array(CFG.edges, dtype=float)


class OrdinalChain(Base):
    """The one that actually fits the structure of the problem.

    The buckets are ORDERED, and multiclass softmax throws that away. Here one
    binary model per edge answers a question the business asks in plain words:
        "will the premium be in by day 7?  by 15?  by 30?  by grace-end?"
    Differencing the (monotonised) cumulative curve gives the bucket
    distribution back. Every sub-model sees all rows -- no shrinking training
    set like the hurdle -- and P(lapse) = 1 - F(grace) is a chain-consistent
    tail rather than a separately fitted head that can disagree with stage 2.
    """
    name = "ordinal_chain"

    def fit(self, train, valid=None):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        self.cols = feature_columns(train)
        assert_no_leakage(self.cols)
        d = train["days_to_pay"]
        self.models = []
        for e in EDGES:
            y = (d <= e).fillna(False).astype(int)
            m = LGBMClassifier(objective="binary", n_estimators=1200,
                               learning_rate=0.04, num_leaves=63,
                               min_child_samples=80, subsample=0.85,
                               subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=5.0, n_jobs=-1, random_state=CFG.seed,
                               verbosity=-1)
            kw = {}
            if valid is not None and len(valid):
                yv = (valid["days_to_pay"] <= e).fillna(False).astype(int)
                kw = dict(eval_set=[(valid[self.cols], yv)],
                          callbacks=[early_stopping(80, verbose=False),
                                     log_evaluation(0)])
            m.fit(train[self.cols], y, categorical_feature=CATEGORICAL, **kw)
            self.models.append(m)
        return self

    def _raw_proba(self, df):
        F = np.column_stack([m.predict_proba(df[self.cols])[:, 1] for m in self.models])
        return from_cdf(F)

    def predict_proba(self, df):
        return self._apply_calibration(self._raw_proba(df))
