"""Two OLS models: the one people actually fit, and the one they should.

`NaiveOLS`: every column thrown at rent in levels. Collinear pair included,
              no transform, default (non-robust) standard errors. This is not a
              straw man; it is what a first pass usually looks like.
`SpecifiedOLS`: the same data after the diagnostics have been read and acted
              on: log target, log area, the collinear twin dropped, the U-shaped
              age term made explicit, HC3 robust errors.

The gap between them is the project's thesis. Both are linear models. Only one
of them is wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rent_price_explainer.features.build import design_matrix

TARGET = "monthly_rent"


class _OLSBase:
    name = "ols"
    log_target = False

    def __init__(self, cov_type: str = "nonrobust"):
        self.cov_type = cov_type
        self.res = None
        self.cols: list[str] = []

    # -------------------------------------------------------------- interface
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, df: pd.DataFrame) -> "_OLSBase":
        X = self._prepare(df)
        self.cols = list(X.columns)
        y = np.log(df[TARGET]) if self.log_target else df[TARGET]
        self.res = sm.OLS(y.astype(float), sm.add_constant(X, has_constant="add")
                          ).fit(cov_type=self.cov_type)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df).reindex(columns=self.cols, fill_value=0.0)
        pred = self.res.predict(sm.add_constant(X, has_constant="add"))
        if not self.log_target:
            return np.asarray(pred, float)
        # Retransforming a log model with exp() alone under-predicts the mean by
        # exp(sigma^2/2) -- Duan's smearing correction, applied honestly.
        smear = float(np.mean(np.exp(self.res.resid)))
        return np.asarray(np.exp(pred), float) * smear

    # ---------------------------------------------------------------- reporting
    def coefficients(self) -> pd.DataFrame:
        r = self.res
        out = pd.DataFrame({
            "coef": r.params, "std_err": r.bse, "t": r.tvalues, "p_value": r.pvalues,
            "ci_low": r.conf_int()[0], "ci_high": r.conf_int()[1],
        })
        out["significant"] = out["p_value"] < 0.05
        return out.round(5)

    @property
    def design(self):
        return self.res.model.exog


class NaiveOLS(_OLSBase):
    """Rent in levels, every feature included, collinear twin and all."""
    name = "naive_ols"
    log_target = False

    def _prepare(self, df):
        return design_matrix(df, log_area=False, drop_collinear=False,
                             add_age_curve=False, include_junk=True)


class SpecifiedOLS(_OLSBase):
    """The same linear model, specified from what the diagnostics said."""
    name = "specified_ols"
    log_target = True

    def __init__(self):
        super().__init__(cov_type="HC3")   # heteroscedasticity-robust errors

    def _prepare(self, df):
        return design_matrix(df, log_area=True, drop_collinear=True,
                             add_age_curve=True, include_junk=False)


class InteractionOLS(SpecifiedOLS):
    """Specified OLS plus the interaction the GBM's SHAP plot pointed at.

    This is the model that ends the argument: it matches or beats the tree on
    accuracy AND still reports an elasticity with a confidence interval. The
    tree was never the better model here -- it was the better *detective*.
    """
    name = "interaction_ols"

    def _prepare(self, df):
        return design_matrix(df, log_area=True, drop_collinear=True,
                             add_age_curve=True, include_junk=False,
                             add_interaction=True)
