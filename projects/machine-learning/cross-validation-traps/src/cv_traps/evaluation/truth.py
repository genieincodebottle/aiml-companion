"""The thing every scheme is trying to estimate.

A cross-validation score is an estimate. Calling one scheme better than another
means comparing both against what they estimate, and on real data you cannot,
which is why the argument is usually settled by assertion.

Deployment for this panel means two things at once: customers the model has
never seen, in periods that had not happened at training time. So the truth is
measured on exactly that, a holdout of unseen customers in future periods, and
every scheme in the project is scored by its distance from it.

Getting this wrong invalidates the whole project, so it is deliberately the
smallest file here and does only one thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cv_traps.config import CFG, Config


def split_panel(df: pd.DataFrame, cfg: Config = CFG
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (development, truth_holdout).

    The holdout is the last `holdout_periods` periods, restricted to customers
    held out entirely from development. Both conditions matter: future periods
    alone would still let the model recognise returning customers, and unseen
    customers alone would still let it validate against a world it has already
    seen.
    """
    rng = np.random.default_rng(cfg.seed + 977)
    customers = np.sort(df["customer_id"].unique())
    held = set(rng.choice(
        customers, size=int(len(customers) * cfg.holdout_customer_share),
        replace=False).tolist())

    cutoff = int(df["period"].max()) - cfg.holdout_periods + 1
    is_future = df["period"] >= cutoff
    is_held = df["customer_id"].isin(held)

    development = df[~is_future & ~is_held].reset_index(drop=True)
    holdout = df[is_future & is_held].reset_index(drop=True)

    if holdout.empty or development.empty:
        raise ValueError(
            "the truth split produced an empty side; check holdout_periods "
            "and holdout_customer_share against the size of the panel")
    if set(development["customer_id"]) & set(holdout["customer_id"]):
        raise AssertionError(
            "a customer appears in both development and the truth holdout, so "
            "the holdout no longer measures unseen customers")
    return development, holdout


def summarise(development: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    return {
        "development_rows": len(development),
        "development_customers": int(development["customer_id"].nunique()),
        "development_periods": sorted(development["period"].unique().tolist()),
        "holdout_rows": len(holdout),
        "holdout_customers": int(holdout["customer_id"].nunique()),
        "holdout_periods": sorted(holdout["period"].unique().tolist()),
        "development_churn_rate": round(float(development["churned"].mean()), 4),
        "holdout_churn_rate": round(float(holdout["churned"].mean()), 4),
    }
