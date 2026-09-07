"""Synthetic rental market with a KNOWN ground truth.

This is the pedagogical core of the project. Real listing data can tell you
which model predicts better; only a generated market can tell you which model
recovered the *true* relationship, because here we wrote it down.

The data-generating process is deliberately built to violate, one at a time,
each OLS assumption the course teaches:

  1. FUNCTIONAL FORM   price is log-linear in the drivers, so a naive
                       level-on-level OLS is mis-specified from line one.
  2. HETEROSCEDASTICITY spread grows with size. Cheap studios vary by a little,
                       luxury flats by a lot. Coefficients stay unbiased; the
                       standard errors (and every p-value built on them) do not.
  3. MULTICOLLINEARITY  `carpet_area` is ~0.999 correlated with `builtup_area`,
                       and `bedrooms` tracks both. R² will not notice. The
                       individual coefficients will be wrecked.
  4. NON-LINEARITY      `age_years` has a U-shape (new build premium, mid-life
                       dip, heritage premium) that no straight line can express.
  5. INFLUENTIAL POINTS a small penthouse cluster with extreme leverage.
  6. INTERACTION       the metro-distance penalty is steeper in premium
                       localities. Main-effects OLS is blind to it by
                       construction; a GBM discovers it unaided.

TRUE_BETAS below are the answer key. Nothing downstream is allowed to import
them for fitting. Only `evaluation/recovery.py` reads them, to score how close
each model got.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- answer key
# Effects on LOG monthly rent, i.e. approximately "percent change per unit".
TRUE_BETAS: dict[str, float] = {
    "intercept": 5.65,
    "log_builtup_area": 0.62,     # 10% more area -> ~6.2% more rent
    "bedrooms": 0.045,
    "bathrooms": 0.035,
    "floor": 0.004,
    "has_lift": 0.055,
    "has_parking": 0.070,
    "furnished_semi": 0.090,
    "furnished_full": 0.185,
    "metro_km": -0.048,           # each km from a metro station costs ~4.7%
    "school_rating": 0.028,
    "crime_index": -0.031,
    # INTERACTION: distance from a metro costs MORE in premium localities,
    # where tenants are paying for commute convenience in the first place.
    # A main-effects-only linear model cannot see this unless told; a tree
    # finds it for free. That asymmetry is the point of the whole comparison.
    "metro_km_x_premium": -0.038,
}
PREMIUM_LOCALITIES = {"tech_park", "riverside"}
# The U-shape in age: rent = ... + AGE_QUAD * (age - AGE_MIN)^2 (on log scale)
AGE_QUAD, AGE_MIN = 0.00022, 28.0
# Multiplicative noise floor and the heteroscedasticity slope.
NOISE_BASE, NOISE_AREA_SLOPE = 0.055, 0.085

LOCALITIES = ["riverside", "old_town", "tech_park", "airport_rd",
              "university", "industrial", "hill_view"]
LOCALITY_EFFECT = {"riverside": 0.155, "old_town": 0.045, "tech_park": 0.190,
                   "airport_rd": -0.030, "university": 0.075,
                   "industrial": -0.115, "hill_view": 0.120}


def generate(n: int = 6000, seed: int = 42, junk_features: int = 8) -> pd.DataFrame:
    """One row per listing. `junk_features` pure-noise columns are included on
    purpose: they are what makes the R-squared-always-rises demonstration
    honest."""
    rng = np.random.default_rng(seed)

    # --- size: the primary driver, lognormal so it is right-skewed like reality
    builtup = np.round(np.exp(rng.normal(6.95, 0.42, n)))          # ~450-3000 sqft
    builtup = np.clip(builtup, 280, 6500)

    # A small penthouse cluster, sized up here so that every downstream column
    # (carpet area included) is derived from the FINAL area. Deriving the twin
    # before this bump would quietly break the collinearity we depend on.
    lux = rng.random(n) < 0.012
    builtup[lux] = np.round(builtup[lux] * rng.uniform(1.8, 3.1, lux.sum()))

    # --- MULTICOLLINEARITY, planted deliberately -------------------------------
    # carpet area is a near-deterministic function of builtup (a loading factor
    # of ~0.80 plus small noise). Any regression given both cannot tell them
    # apart, and will hand you two wild, unstable coefficients instead of one
    # sensible coefficient.
    carpet = builtup * rng.normal(0.80, 0.018, n)

    bedrooms = np.clip(np.round(builtup / 520 + rng.normal(0, 0.45, n)), 1, 6)
    bathrooms = np.clip(np.round(bedrooms * 0.75 + rng.normal(0, 0.35, n)), 1, 5)

    floor = rng.integers(0, 22, n)
    has_lift = ((floor > 3) | (rng.random(n) < 0.45)).astype(int)
    has_parking = (rng.random(n) < 0.62).astype(int)

    furnishing = rng.choice(["unfurnished", "semi", "full"], n, p=[0.42, 0.38, 0.20])
    locality = rng.choice(LOCALITIES, n,
                          p=[0.14, 0.13, 0.20, 0.12, 0.15, 0.11, 0.15])

    metro_km = np.round(np.clip(rng.gamma(2.2, 1.35, n), 0.1, 14.0), 2)
    school_rating = np.clip(np.round(rng.normal(6.4, 1.6, n), 1), 1, 10)
    crime_index = np.clip(np.round(rng.normal(4.8, 1.9, n), 1), 0.5, 10)

    # --- NON-LINEARITY: U-shaped age effect ------------------------------------
    age_years = np.clip(np.round(rng.gamma(2.6, 6.0, n)), 0, 85)

    b = TRUE_BETAS
    log_rent = (
        b["intercept"]
        + b["log_builtup_area"] * np.log(builtup)
        + b["bedrooms"] * bedrooms
        + b["bathrooms"] * bathrooms
        + b["floor"] * floor
        + b["has_lift"] * has_lift
        + b["has_parking"] * has_parking
        + b["furnished_semi"] * (furnishing == "semi")
        + b["furnished_full"] * (furnishing == "full")
        + b["metro_km"] * metro_km
        + b["school_rating"] * school_rating
        + b["crime_index"] * crime_index
        + AGE_QUAD * (age_years - AGE_MIN) ** 2
        + np.array([LOCALITY_EFFECT[x] for x in locality])
        + b["metro_km_x_premium"] * metro_km
        * np.array([x in PREMIUM_LOCALITIES for x in locality], dtype=float)
    )

    # --- HETEROSCEDASTICITY: noise scales with size ----------------------------
    sigma = NOISE_BASE + NOISE_AREA_SLOPE * (np.log(builtup) - np.log(builtup).mean())
    sigma = np.clip(sigma, 0.03, None)
    log_rent = log_rent + rng.normal(0, sigma)

    # --- INFLUENTIAL OUTLIERS: the penthouse cluster also carries a price
    # premium beyond its size, which is what gives it high leverage.
    log_rent[lux] += rng.normal(0.75, 0.20, int(lux.sum()))

    df = pd.DataFrame({
        "listing_id": [f"L{i:06d}" for i in range(n)],
        "builtup_area": builtup.astype(float),
        "carpet_area": np.round(carpet, 1),
        "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int),
        "floor": floor.astype(int),
        "age_years": age_years.astype(int),
        "has_lift": has_lift,
        "has_parking": has_parking,
        "furnishing": furnishing,
        "locality": locality,
        "metro_km": metro_km,
        "school_rating": school_rating,
        "crime_index": crime_index,
        "monthly_rent": np.round(np.exp(log_rent), -1),
    })

    # --- pure noise, for the R² demonstration ---------------------------------
    for j in range(junk_features):
        df[f"junk_{j+1}"] = np.round(rng.normal(0, 1, n), 4)

    return df


def true_coefficients() -> pd.Series:
    """The answer key, as a Series. Used ONLY for scoring recovery."""
    return pd.Series(TRUE_BETAS, name="true_beta")
