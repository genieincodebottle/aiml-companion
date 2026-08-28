"""The five OLS assumption tests, each returning a verdict rather than a p-value.

A p-value nobody interprets is not a diagnostic. Every function here returns a
`Check` with a pass/fail, the statistic, and what the
failure actually costs you. Some failures bias your coefficients; others leave
the coefficients fine and only wreck the standard errors. Treating those two as
the same problem is the most common mistake in applied regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


@dataclass
class Check:
    name: str
    passed: bool
    statistic: float
    p_value: float | None
    consequence: str          # what it costs you if it fails
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        p = f"p={self.p_value:.2e}" if self.p_value is not None else "--"
        return f"[{mark}] {self.name:22s} stat={self.statistic:>10.4f}  {p}"


# --------------------------------------------------------------------- 1. VIF
def vif(X: pd.DataFrame, threshold: float = 5.0) -> tuple[Check, pd.DataFrame]:
    """Variance Inflation Factor: how much collinearity inflates each
    coefficient's variance. VIF=10 means that coefficient's standard error is
    sqrt(10) ~ 3.2x larger than it would be with independent predictors.

    Crucially: collinearity does NOT hurt predictions. It destroys the
    *interpretation*, which is the entire reason you chose a linear model.
    """
    num = X.select_dtypes(include=[np.number]).astype(float)
    num = num.loc[:, num.std() > 0]
    Xc = np.column_stack([np.ones(len(num)), num.to_numpy()])
    vals = [variance_inflation_factor(Xc, i + 1) for i in range(num.shape[1])]
    table = (pd.DataFrame({"feature": num.columns, "vif": vals})
             .sort_values("vif", ascending=False, ignore_index=True))
    worst = table.iloc[0]
    return Check(
        name="multicollinearity",
        passed=bool(worst["vif"] < threshold),
        statistic=float(worst["vif"]), p_value=None,
        consequence=("coefficients become unstable and can flip sign; "
                     "predictions are unaffected, interpretation is destroyed"),
        detail={"worst_feature": worst["feature"],
                "n_above_threshold": int((table["vif"] >= threshold).sum())},
    ), table


def condition_number(X: pd.DataFrame) -> Check:
    """Overall conditioning of the design matrix. >30 is the classic warning."""
    num = X.select_dtypes(include=[np.number]).astype(float)
    num = (num - num.mean()) / num.std(ddof=0).replace(0, 1)
    cn = float(np.linalg.cond(num.to_numpy()))
    return Check("condition_number", cn < 30, cn, None,
                 "near-singular design matrix; coefficient estimates are fragile")


# ------------------------------------------------------- 2. heteroscedasticity
def breusch_pagan(resid: np.ndarray, X: pd.DataFrame, alpha: float = 0.05) -> Check:
    """Is the error variance constant across fitted values?

    If not, your coefficients are still unbiased -- but every standard error,
    t-statistic, p-value and confidence interval is wrong. The fix is rarely to
    abandon the model: it is robust (HC3) standard errors, or a transform that
    stabilises the variance.
    """
    exog = np.column_stack([np.ones(len(X)),
                            X.select_dtypes(include=[np.number]).to_numpy()])
    lm, lm_p, _, _ = het_breuschpagan(resid, exog)
    return Check("heteroscedasticity", bool(lm_p > alpha), float(lm), float(lm_p),
                 "coefficients stay unbiased, but ALL standard errors, "
                 "t-stats and p-values are invalid -> use HC3 robust errors")


# ------------------------------------------------------------- 3. functional form
def reset_test(model, power: int = 3, alpha: float = 0.05) -> Check:
    """Ramsey RESET: are powers of the fitted values significant?

    If they are, the relationship is not linear in the form you specified --
    the model is mis-specified, and the coefficients are biased, not merely
    imprecise. This is the failure that actually invalidates conclusions.
    """
    res = linear_reset(model, power=power, use_f=True)
    p = float(res.pvalue)
    return Check("functional_form", bool(p > alpha), float(res.fvalue), p,
                 "MIS-SPECIFIED: coefficients are biased, not just imprecise "
                 "-- transform the target or add the missing non-linear term")


# ---------------------------------------------------------------- 4. normality
def normality(resid: np.ndarray, alpha: float = 0.05) -> Check:
    """Jarque-Bera on the residuals.

    The least important test of the five, and the most over-weighted. With a
    large sample the CLT carries inference anyway; badly non-normal residuals
    usually point at outliers or a missing transform, which the other tests
    diagnose better.
    """
    jb, p, skew, kurt = jarque_bera(resid)
    return Check("residual_normality", bool(p > alpha), float(jb), float(p),
                 "small-sample inference is unreliable; usually a symptom of "
                 "outliers or a missing transform rather than a problem itself",
                 {"skew": float(skew), "kurtosis": float(kurt)})


# ------------------------------------------------------------ 5. independence
def independence(resid: np.ndarray) -> Check:
    """Durbin-Watson: ~2 means uncorrelated residuals."""
    dw = float(durbin_watson(resid))
    return Check("residual_independence", 1.5 < dw < 2.5, dw, None,
                 "correlated errors understate standard errors "
                 "(matters most for time-ordered or clustered data)")


# ------------------------------------------------------------ influence points
def influence(model, threshold_mult: float = 4.0) -> tuple[Check, pd.DataFrame]:
    """Cook's distance: which rows are single-handedly moving the fit?

    The usual cutoff is 4/n. High-influence rows are not automatically errors --
    a genuine penthouse is real data. The decision to keep, cap or model them
    separately is a modelling choice you must make explicitly.
    """
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    cutoff = threshold_mult / len(cooks)
    flagged = np.where(cooks > cutoff)[0]
    table = (pd.DataFrame({"row": flagged, "cooks_d": cooks[flagged]})
             .sort_values("cooks_d", ascending=False, ignore_index=True))
    return Check("influential_points", len(flagged) == 0, float(cooks.max()), None,
                 "a handful of rows are driving the coefficients; decide "
                 "explicitly whether to keep, cap or model them separately",
                 {"n_flagged": int(len(flagged)), "cutoff": float(cutoff),
                  "share": round(len(flagged) / len(cooks), 4)}), table


# ------------------------------------------------------------------- run them all
def run_all(model, X: pd.DataFrame, resid: np.ndarray | None = None) -> dict:
    """Every check at once. Returns checks plus the VIF and influence tables."""
    resid = model.resid if resid is None else resid
    vif_check, vif_table = vif(X)
    infl_check, infl_table = influence(model)
    checks = [
        reset_test(model),
        breusch_pagan(np.asarray(resid), X),
        vif_check,
        condition_number(X),
        normality(np.asarray(resid)),
        independence(np.asarray(resid)),
        infl_check,
    ]
    return {"checks": checks, "vif_table": vif_table, "influence_table": infl_table,
            "n_failed": sum(not c.passed for c in checks)}


def summary_frame(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([{
        "check": c.name, "verdict": "PASS" if c.passed else "FAIL",
        "statistic": round(c.statistic, 4),
        "p_value": None if c.p_value is None else float(f"{c.p_value:.3e}"),
        "if it fails": c.consequence,
    } for c in checks])
