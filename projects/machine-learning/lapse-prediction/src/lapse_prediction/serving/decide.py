"""The decision layer -- the part the business actually consumes.

A probability is not an action. This turns the bucket distribution into a
retention queue: who to call, when to call them, and what is at stake.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG
from lapse_prediction.evaluation.metrics import expected_days


def score(df: pd.DataFrame, proba: np.ndarray, cfg=CFG,
          capacity_pct: float = 0.20, contact_lead_days: int = 5) -> pd.DataFrame:
    p_lapse = proba[:, cfg.lapse_index]
    eta = expected_days(proba, cfg)

    out = pd.DataFrame({
        "policy_id": df["policy_id"].values,
        "due_date": df["due_date"].values,
        "p_lapse": p_lapse,
        "expected_days_if_paid": np.round(eta, 1),
        "premium_at_risk": df["annual_premium"].values * p_lapse,
    })
    for i, name in enumerate(cfg.class_names):
        out[f"p_{name}"] = proba[:, i]

    # Contact just before the mass of the distribution would arrive anyway --
    # calling someone who was going to pay on day 3 is wasted capacity.
    out["contact_on_day"] = np.clip(
        np.round(eta) - contact_lead_days, 0, cfg.grace_days - 1).astype(int)
    out.loc[out["p_lapse"] > 0.5, "contact_on_day"] = 0

    # Rank by expected rupees saved, not by raw probability.
    out["priority_score"] = out["premium_at_risk"]
    cut = out["priority_score"].quantile(1 - capacity_pct)
    out["action"] = np.where(out["priority_score"] >= cut, "call", "monitor")
    return out.sort_values("priority_score", ascending=False, ignore_index=True)


def value_of_queue(scored: pd.DataFrame, save_rate: float = 0.25) -> dict:
    """Rough business case: premium retained if calling saves `save_rate` of
    the lapses you actually contact."""
    called = scored[scored["action"] == "call"]
    return {
        "policies_called": int(len(called)),
        "share_of_book": round(len(called) / max(len(scored), 1), 3),
        "premium_at_risk_covered": round(float(called["premium_at_risk"].sum()), 0),
        "share_of_total_risk_covered": round(
            float(called["premium_at_risk"].sum()
                  / max(scored["premium_at_risk"].sum(), 1e-9)), 3),
        "expected_premium_saved": round(
            float(called["premium_at_risk"].sum() * save_rate), 0),
    }
