"""Design matrix construction, with each fix switchable.

The switches exist so the notebook can turn one remedy on at a time and show
what it bought, rather than presenting a finished specification and asking the
reader to take it on faith.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "monthly_rent"
ID = "listing_id"
CATEGORICAL = ["furnishing", "locality"]
# Dummy baselines, chosen EXPLICITLY. pandas' drop_first drops whichever level
# sorts first alphabetically, which silently made "full" the furnishing
# reference -- so every furnishing coefficient was reported relative to the
# most expensive level instead of the cheapest, flipping its sign. A reference
# level is a modelling decision and belongs in the open.
REFERENCE_LEVEL = {"furnishing": "unfurnished", "locality": "old_town"}
# carpet_area is ~0.97 correlated with builtup_area by construction. Keeping
# both is the multicollinearity the diagnostics are meant to catch.
COLLINEAR_TWIN = "carpet_area"


def junk_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("junk_")]


PREMIUM_LOCALITIES = {"tech_park", "riverside"}


def design_matrix(df: pd.DataFrame, *, log_area: bool = True,
                  drop_collinear: bool = True, add_age_curve: bool = True,
                  include_junk: bool = False,
                  add_interaction: bool = False) -> pd.DataFrame:
    """Build the model matrix. Every argument corresponds to one diagnostic fix.

    log_area        -> the RESET test said the level-on-level form is wrong
    drop_collinear  -> VIF said carpet_area and builtup_area are the same column
    add_age_curve   -> the residual-vs-age plot showed a U, so model it
    include_junk    -> only for the R²-inflation demonstration
    add_interaction -> the GBM's SHAP interaction plot revealed that the metro
                       penalty is steeper in premium localities. A linear model
                       can represent that perfectly well -- it just has to be
                       TOLD, which is the real difference between the families.
    """
    X = df.drop(columns=[c for c in (TARGET, ID) if c in df.columns]).copy()

    if not include_junk:
        X = X.drop(columns=junk_columns(X), errors="ignore")

    if drop_collinear:
        X = X.drop(columns=[COLLINEAR_TWIN], errors="ignore")

    if log_area:
        for c in ("builtup_area", "carpet_area"):
            if c in X.columns:
                X[f"log_{c}"] = np.log(X[c])
                X = X.drop(columns=[c])

    if add_age_curve and "age_years" in X.columns:
        # The true effect is quadratic. Centring keeps the linear and squared
        # terms from becoming collinear with each other.
        centred = X["age_years"] - X["age_years"].mean()
        X["age_centred_sq"] = centred ** 2

    if add_interaction and {"metro_km", "locality"} <= set(X.columns):
        X["metro_km_x_premium"] = X["metro_km"] * X["locality"].isin(
            PREMIUM_LOCALITIES).astype(float)

    cats = [c for c in CATEGORICAL if c in X.columns]
    X = pd.get_dummies(X, columns=cats, drop_first=False, dtype=float)
    # drop the chosen reference level by NAME, never by sort order
    X = X.drop(columns=[f"{c}_{REFERENCE_LEVEL[c]}" for c in cats
                        if f"{c}_{REFERENCE_LEVEL[c]}" in X.columns])
    return X.astype(float)


def feature_names(df: pd.DataFrame, **kw) -> list[str]:
    return list(design_matrix(df, **kw).columns)
