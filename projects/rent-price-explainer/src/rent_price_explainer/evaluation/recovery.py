"""Did the model recover the TRUE relationship?

This is the question a real dataset can never answer and the reason this project
generates its own market. Predictive accuracy and parameter recovery are
different objectives, and a model can be excellent at one while being useless at
the other. Here both are measured side by side.

Read `TRUE_BETAS` only from this module. Nothing in the fitting path is allowed
to see the answer key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rent_price_explainer.data.generate import TRUE_BETAS

# design-matrix column -> answer-key name
COEF_MAP = {
    "log_builtup_area": "log_builtup_area",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "floor": "floor",
    "has_lift": "has_lift",
    "has_parking": "has_parking",
    "furnishing_semi": "furnished_semi",
    "furnishing_full": "furnished_full",
    "metro_km": "metro_km",
    "school_rating": "school_rating",
    "crime_index": "crime_index",
    "metro_km_x_premium": "metro_km_x_premium",
}


def score_recovery(model, label: str | None = None) -> pd.DataFrame:
    """Compare a fitted linear model's coefficients with the truth.

    Only meaningful for a model fitted on the LOG target, since the answer key
    is expressed as effects on log rent. A level-target model is reported as
    not comparable rather than silently rescaled. Pretending otherwise would
    be the exact sloppiness this project is about.
    """
    def _not_comparable(note: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "model": label, "term": "(all)", "true_beta": np.nan,
            "estimated": np.nan, "error": np.nan, "pct_error": np.nan,
            "ci_covers_truth": False, "note": note,
        }])

    # A model with no coefficients is not an error case to be handled quietly --
    # it is the trade this project is about, so it is reported in the table.
    if getattr(model, "res", None) is None:
        return _not_comparable(
            "no coefficients exist -- this model cannot state an effect size")
    if not getattr(model, "log_target", False):
        return _not_comparable(
            "model fitted on levels, not log rent -- units are not comparable")

    params = model.res.params
    rows = []
    for col, key in COEF_MAP.items():
        if col not in params.index:
            continue
        true, est = TRUE_BETAS[key], float(params[col])
        ci = model.res.conf_int().loc[col]
        rows.append({
            "term": col,
            "true_beta": true,
            "estimated": round(est, 4),
            "error": round(est - true, 4),
            "pct_error": round(100 * (est - true) / abs(true), 1),
            "ci_covers_truth": bool(ci[0] <= true <= ci[1]),
        })
    out = pd.DataFrame(rows)
    out.insert(0, "model", label if label else "")
    return out


def recovery_summary(recovery: pd.DataFrame) -> dict:
    """One line per model: how close, and how often the interval was honest."""
    if recovery["true_beta"].isna().all():
        return {"comparable": False}
    return {
        "comparable": True,
        "mean_abs_pct_error": round(float(recovery["pct_error"].abs().mean()), 1),
        "max_abs_pct_error": round(float(recovery["pct_error"].abs().max()), 1),
        "worst_term": recovery.loc[recovery["pct_error"].abs().idxmax(), "term"],
        "ci_coverage": round(float(recovery["ci_covers_truth"].mean()), 3),
        "n_terms": int(len(recovery)),
    }


def collinearity_damage(df, n_seeds: int = 8) -> pd.DataFrame:
    """Refit on bootstrap resamples and watch the collinear coefficients swing.

    A single fit hides this: you get one number and it looks authoritative. The
    spread across resamples is what multicollinearity actually does to you, and
    it is visible only when you look for it.
    """
    from rent_price_explainer.features.build import design_matrix
    import statsmodels.api as sm

    rows = []
    for seed in range(n_seeds):
        boot = df.sample(len(df), replace=True, random_state=seed)
        for keep_twin in (True, False):
            X = design_matrix(boot, log_area=True, drop_collinear=not keep_twin,
                              add_age_curve=True, include_junk=False)
            res = sm.OLS(np.log(boot["monthly_rent"]),
                         sm.add_constant(X, has_constant="add")).fit()
            if "log_builtup_area" in res.params.index:
                rows.append({"seed": seed,
                             "spec": "both areas kept" if keep_twin else "twin dropped",
                             "log_builtup_area": round(float(res.params["log_builtup_area"]), 4)})
    out = pd.DataFrame(rows)
    return (out.groupby("spec")["log_builtup_area"]
            .agg(["mean", "std", "min", "max"]).round(4)
            .assign(true=TRUE_BETAS["log_builtup_area"]).reset_index())
