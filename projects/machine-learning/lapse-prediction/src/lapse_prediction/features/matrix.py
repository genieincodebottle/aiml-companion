"""Shared numeric design matrix for the models that can't eat pandas categoricals
(logistic regression, Cox, the neural net). Fitted on train, applied everywhere."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.features.build import CATEGORICAL, feature_columns


class Matrix:
    def __init__(self, standardize: bool = True, drop_first: bool = False):
        self.standardize = standardize
        self.drop_first = drop_first
        self.cols = self.dummies = self.med = self.mu = self.sd = None

    def fit(self, df: pd.DataFrame):
        self.cols = feature_columns(df)
        X = self._raw(df)
        self.dummies = X.columns
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu, self.sd = Xf.mean(), Xf.std().replace(0, 1.0)
        return self

    def _raw(self, df):
        X = df[self.cols].copy()
        X = pd.get_dummies(X, columns=[c for c in CATEGORICAL if c in X.columns],
                           dummy_na=False, drop_first=self.drop_first, dtype=float)
        return X.astype(float)

    def transform(self, df) -> np.ndarray:
        X = self._raw(df).reindex(columns=self.dummies, fill_value=0.0)
        X = X.fillna(self.med)
        if self.standardize:
            X = (X - self.mu) / self.sd
        return X.to_numpy(dtype=np.float64)
