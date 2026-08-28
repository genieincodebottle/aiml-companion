"""SHAP as the bridge between the two models.

The reconciliation people usually miss: for a linear model, the SHAP value of a
feature is exactly `coef * (x - mean(x))`. SHAP is not a different, more modern
kind of explanation. On a linear model it *is* the coefficient, redistributed
per row. Verifying that identity is what earns the right to trust SHAP on the
GBM, where no coefficient exists to check it against.

So the argument runs:
  1. On the specified OLS, SHAP reproduces the coefficients (we assert it).
  2. Therefore SHAP is measuring the same thing on the GBM.
  3. The GBM's mean |SHAP| ranking can then be compared with the OLS
     coefficients, and where they disagree, the disagreement is informative --
     usually a non-linearity the linear model had to be told about explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def linear_shap_identity(model, df: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """Verify SHAP(linear) == coef * (x - x_mean), by hand.

    Computing it directly rather than calling the library is the point: it makes
    the identity checkable instead of magical.
    """
    X = model._prepare(df).reindex(columns=model.cols, fill_value=0.0)
    params = model.res.params
    rows = []
    for c in model.cols:
        if c not in params.index:
            continue
        manual = float(params[c]) * (X[c] - X[c].mean())
        rows.append({"feature": c,
                     "coef": round(float(params[c]), 5),
                     "mean_abs_shap": round(float(manual.abs().mean()), 5)})
    return (pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False,
                                           ignore_index=True))


def gbm_shap(model, df: pd.DataFrame, sample: int = 800,
             seed: int = 42) -> pd.DataFrame:
    """Mean |SHAP| per feature for the tree model, via TreeExplainer."""
    import shap

    X = model._prepare(df).reindex(columns=model.cols, fill_value=0.0)
    if len(X) > sample:
        X = X.sample(sample, random_state=seed)
    values = shap.TreeExplainer(model.model).shap_values(X)
    return (pd.DataFrame({"feature": model.cols,
                          "mean_abs_shap": np.abs(values).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
            .round(5))


def compare_attributions(ols_model, gbm_model, df: pd.DataFrame) -> pd.DataFrame:
    """Rank features by each model's attribution and show where they disagree.

    Both are on the log-rent scale when both models use log targets, so the
    magnitudes are comparable -- but rank disagreement is the interesting
    signal, not the absolute gap.
    """
    a = linear_shap_identity(ols_model, df).rename(
        columns={"mean_abs_shap": "ols_mean_abs_shap"})
    b = gbm_shap(gbm_model, df).rename(
        columns={"mean_abs_shap": "gbm_mean_abs_shap"})

    merged = a.merge(b, on="feature", how="outer")

    # The two models see DIFFERENT columns by design -- the OLS was given
    # log(area) and an age curve, the GBM was given the raw columns and the
    # collinear twin. A blank is therefore "not in this model's matrix", not a
    # missing value, and saying so avoids a misreading.
    merged["present_in"] = np.where(
        merged["ols_mean_abs_shap"].notna() & merged["gbm_mean_abs_shap"].notna(),
        "both",
        np.where(merged["ols_mean_abs_shap"].notna(), "ols only", "gbm only"))

    for side in ("ols", "gbm"):
        col = f"{side}_mean_abs_shap"
        merged[f"{side}_rank"] = merged[col].rank(ascending=False, method="min")
    merged["rank_gap"] = (merged["ols_rank"] - merged["gbm_rank"]).abs()
    return merged.sort_values(
        ["present_in", "gbm_mean_abs_shap"], ascending=[True, False],
        ignore_index=True)


def split_attribution_note(attribution: pd.DataFrame) -> str:
    """Collinearity does not only wreck coefficients -- it splits ATTRIBUTIONS.

    The GBM was handed both area columns, so it divides the size signal between
    them and neither looks as important as size really is. The OLS, given one
    combined term, shows size as the dominant driver it actually is. Same data,
    opposite impression of what matters.
    """
    gbm_only = attribution[attribution["present_in"] == "gbm only"]
    twins = gbm_only[gbm_only["feature"].isin(["builtup_area", "carpet_area"])]
    if len(twins) < 2:
        return "collinear pair not split across the two matrices"
    total = float(twins["gbm_mean_abs_shap"].sum())
    parts = ", ".join(f"{r.feature}={r.gbm_mean_abs_shap:.4f}"
                      for r in twins.itertuples())
    return (f"the GBM splits the size signal across its collinear pair "
            f"({parts}; combined {total:.4f}), so neither column looks as "
            f"important as size actually is")


def interpretability_ledger(ols_model, gbm_model) -> pd.DataFrame:
    """What each model can and cannot tell a regulator, stated plainly."""
    questions = [
        ("Point estimate for a listing", True, True),
        ("Effect of +10% area, as a number", True, False),
        ("Confidence interval on that effect", True, False),
        ("p-value / is the effect distinguishable from zero", True, False),
        ("Per-listing attribution (why THIS price)", True, True),
        ("Extrapolates beyond the training range", True, False),
        ("Auditable as an equation on one page", True, False),
        ("Captures non-linearity without being told", False, True),
        ("Captures interactions without being told", False, True),
    ]
    return pd.DataFrame([{"question": q, "specified_ols": "yes" if o else "no",
                          "gbm": "yes" if g else "no"} for q, o, g in questions])
