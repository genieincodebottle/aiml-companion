"""A churn panel built so that every cross-validation trap has a known answer.

Cross-validation returns a number. The number is usually optimistic, and the
interesting question is not "is CV biased" but "by how much, and which of my
choices caused it". That question is unanswerable on real data, because you
never see the truth you are trying to estimate.

Here you do. The panel has a deliberate structure that each trap exploits:

  * a latent per-customer effect `u`, learnable only if the same customer
    appears on both sides of a split  ->  the GROUP trap
  * coefficients and a base rate that drift across periods  ->  the TIME trap
  * many pure-noise columns, so selecting features on all the data before
    validating finds noise that happens to fit  ->  the PREPROCESSING trap
  * enough label noise that candidate models are genuinely close, so picking
    the best by CV mostly picks the luckiest  ->  the SELECTION trap
  * a tunable per-fold sample size, so a single CV number can be shown to move
    more between repeats than between models  ->  the VARIANCE trap

The truth every scheme is scored against is defined in `evaluation/truth.py`:
unseen customers in future periods, which is what deployment actually is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Features that are constant (or nearly so) within a customer. Together they
#: fingerprint the customer, which is precisely what lets a flexible model
#: recover the latent effect `u` when a split puts the same customer in both
#: train and test. Remove these and the group trap quietly stops working.
CUSTOMER_FEATURES = ("signup_score", "plan_price", "region_index",
                     "baseline_usage")

#: Features that vary period to period within a customer. These carry the
#: honest, transferable signal.
PERIOD_FEATURES = ("monthly_spend", "sessions", "support_tickets",
                   "days_since_login", "discount_pct")

ID_COLUMNS = ("customer_id", "period", "row_id")
TARGET = "churned"


def _noise_columns(n_noise: int) -> list[str]:
    return [f"noise_{i:03d}" for i in range(n_noise)]


def feature_names(n_noise: int) -> list[str]:
    return list(CUSTOMER_FEATURES) + list(PERIOD_FEATURES) + _noise_columns(n_noise)


def generate(n_customers: int = 2600, n_periods: int = 12,
             n_noise: int = 150, seed: int = 42,
             group_effect: float = 2.50, drift: float = 0.90,
             missing_rate: float = 0.08,
             label_noise: float = 0.05) -> pd.DataFrame:
    """Build the panel. One row per customer-period.

    Args:
        group_effect: standard deviation of the latent per-customer effect. At
            0.0 customers are exchangeable and grouped CV buys you nothing,
            which is the control worth running.
        drift: how far the coefficients travel from the first period to the
            last. At 0.0 the process is stationary and random KFold is a fair
            estimator of future performance.
        label_noise: probability of flipping the drawn label. Pushes the
            candidate models close enough together that CV ranking becomes
            mostly luck, which is what the selection trap needs.
    """
    if n_customers < 200:
        raise ValueError(
            f"n_customers={n_customers} is too small for grouped folds to hold "
            "a usable number of customers each; use at least 200")
    if n_periods < 4:
        raise ValueError(
            "n_periods must be at least 4 so forward-chaining validation has "
            "somewhere to chain to")

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- customers
    u = rng.normal(0.0, group_effect, n_customers)      # the latent effect
    signup_score = rng.normal(600, 120, n_customers)
    plan_price = rng.choice([9.0, 19.0, 49.0, 99.0], n_customers,
                            p=[0.34, 0.36, 0.22, 0.08])
    region_index = rng.integers(0, 14, n_customers).astype(float)
    baseline_usage = rng.gamma(3.0, 40.0, n_customers)

    # Each customer is observed over a contiguous run of periods, the way a
    # real subscription book is: people join and leave, they are not a
    # rectangle.
    start = rng.integers(0, n_periods - 2, n_customers)
    span = rng.integers(3, n_periods + 1, n_customers)

    rows = []
    for c in range(n_customers):
        last = min(n_periods, start[c] + span[c])
        for p in range(start[c], last):
            rows.append((c, p))
    cust_idx = np.array([r[0] for r in rows])
    period = np.array([r[1] for r in rows], dtype=float)
    n = len(rows)

    # ------------------------------------------------------ period features
    # Scales differ by three orders of magnitude on purpose, so a model that
    # needs scaling actually needs it.
    monthly_spend = (plan_price[cust_idx] * rng.gamma(2.0, 1.1, n)
                     + baseline_usage[cust_idx] * 0.35)
    sessions = rng.poisson(np.clip(baseline_usage[cust_idx] / 12.0, 0.3, 40), n)
    support_tickets = rng.poisson(0.45 + 0.02 * period, n)
    days_since_login = rng.gamma(2.0, 6.0, n) + 0.8 * support_tickets
    discount_pct = rng.beta(1.4, 9.0, n) * 100.0

    # ------------------------------------------------------------- the truth
    # Two coefficients travel linearly across the panel. `t` is 0 at the first
    # period and 1 at the last, so `drift` reads directly as "how far".
    t = period / max(n_periods - 1, 1)
    # Coefficients are deliberately large. An earlier draft used values a
    # quarter of these and the whole panel sat at AUC 0.55, because the latent
    # effect `u` was bigger than every observable coefficient combined: the
    # unlearnable part drowned the learnable one and no trap could be measured
    # against a truth that was itself indistinguishable from chance.
    b_login = 0.1375 + drift * 0.1375 * t        # matters more over time
    b_tickets = 1.00 - drift * 0.75 * t          # matters less over time
    intercept = -1.10 - drift * 0.85 * t         # the base rate drifts too

    z = (intercept
         + b_login * (days_since_login - 12.0)
         + b_tickets * (support_tickets - 0.6)
         - 0.0275 * (monthly_spend - 60.0)
         - 0.0040 * (signup_score[cust_idx] - 600.0)
         + 0.0300 * (discount_pct - 12.0)
         - 0.0750 * (sessions - 8.0)
         + u[cust_idx])                          # the part only grouping hides

    p_churn = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p_churn).astype(int)
    flip = rng.random(n) < label_noise
    y = np.where(flip, 1 - y, y)

    data = {
        "customer_id": cust_idx,
        "period": period.astype(int),
        "signup_score": signup_score[cust_idx],
        "plan_price": plan_price[cust_idx],
        "region_index": region_index[cust_idx],
        "baseline_usage": baseline_usage[cust_idx],
        "monthly_spend": monthly_spend,
        "sessions": sessions.astype(float),
        "support_tickets": support_tickets.astype(float),
        "days_since_login": days_since_login,
        "discount_pct": discount_pct,
    }
    # Pure noise. None of it is in `z`, so any apparent usefulness is an
    # artefact of how it was measured.
    noise = rng.normal(0.0, 1.0, (n, n_noise))
    for i, name in enumerate(_noise_columns(n_noise)):
        data[name] = noise[:, i]

    df = pd.DataFrame(data)
    df.insert(0, "row_id", np.arange(n))
    df[TARGET] = y

    # Missingness, so an imputer is genuinely required rather than decorative.
    if missing_rate > 0:
        for col in ("monthly_spend", "days_since_login", "discount_pct"):
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan

    return df.sort_values(["period", "customer_id"], ignore_index=True)
