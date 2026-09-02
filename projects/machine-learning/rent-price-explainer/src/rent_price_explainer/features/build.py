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
                  add_interaction: bool = False,
                  age_center: float | None = None) -> pd.DataFrame:
    """Build the model matrix. Every argument corresponds to one diagnostic fix.

    log_area        -> the RESET test said the level-on-level form is wrong
    drop_collinear  -> VIF said carpet_area and builtup_area are the same column
    add_age_curve   -> the residual-vs-age plot showed a U, so model it
    include_junk    -> only for the R²-inflation demonstration
    add_interaction -> the GBM's SHAP interaction plot revealed that the metro
                       penalty is steeper in premium localities. A linear model
                       can represent that perfectly well -- it just has to be
                       TOLD, which is the real difference between the families.
    age_center      -> the constant the age curve is centred on. This is a
                       FITTED PARAMETER, not a property of the rows in front of
                       you, and it must be frozen on the training set and
                       reused at inference. Pass it explicitly; `None` means
                       "learn it from this frame", which is only ever correct
                       on the training frame itself. See the note below.
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
        #
        # The centring constant is LEARNED FROM DATA, which makes it exactly as
        # dangerous as a fitted scaler. Recomputing it on whatever rows you are
        # scoring makes the design matrix depend on the batch: the same listing
        # priced alone and priced inside a batch of 1,500 gets a different
        # `age_centred_sq`, and therefore a different rent. That is not a
        # rounding difference -- it moved predictions by up to 3.9% here, and a
        # single-row request centres age on itself, sending the term to exactly
        # 0 every time. Freeze it at fit and pass it back in.
        centre = (float(X["age_years"].mean()) if age_center is None
                  else float(age_center))
        X["age_centred_sq"] = (X["age_years"] - centre) ** 2

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


def learn_age_center(df: pd.DataFrame) -> float:
    """The one place the age centring constant is estimated.

    Named rather than inlined so that "this is fitted state" is visible at the
    call site, and so a model can store it next to its coefficients.
    """
    return float(df["age_years"].mean())
