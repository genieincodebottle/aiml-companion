"""The gradient-boosting challenger.

Fit on the same rows and the same raw columns, with none of the specification
work: no log transform, no collinearity surgery, no age curve. That is the
honest comparison, because it is exactly why people reach for a GBM: it finds
the shape itself.

What it cannot do is hand you an elasticity. `evaluation/recovery.py` scores
both models against the known true coefficients, and that is where the trade
becomes concrete rather than rhetorical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rent_price_explainer.features.build import TARGET, design_matrix


class GBM:
    name = "gbm"

    def __init__(self, log_target: bool = True, seed: int = 42, **kw):
        self.log_target = log_target
        defaults = dict(max_iter=500, learning_rate=0.06, max_depth=None,
                        max_leaf_nodes=31, min_samples_leaf=25,
                        l2_regularization=1.0, early_stopping=True,
                        validation_fraction=0.15, random_state=seed)
        self.params = {**defaults, **kw}   # caller overrides win
        self.model = None
        self.cols: list[str] = []

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Deliberately NO specification work: raw columns, collinear twin kept.
        # Trees are indifferent to collinearity and monotone transforms, which
        # is the whole point of the comparison.
        return design_matrix(df, log_area=False, drop_collinear=False,
                             add_age_curve=False, include_junk=False)

    def fit(self, df: pd.DataFrame) -> "GBM":
        X = self._prepare(df)
        self.cols = list(X.columns)
        y = np.log(df[TARGET]) if self.log_target else df[TARGET]
        self.model = HistGradientBoostingRegressor(**self.params).fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df).reindex(columns=self.cols, fill_value=0.0)
        pred = self.model.predict(X)
        return np.exp(pred) if self.log_target else pred

    def coefficients(self) -> pd.DataFrame:
        """There aren't any. Stated explicitly rather than left as an absence --
        this is the cost side of the accuracy trade."""
        return pd.DataFrame(columns=["coef", "std_err", "t", "p_value"])
