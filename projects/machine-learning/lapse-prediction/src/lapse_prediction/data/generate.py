"""Synthetic generator that mimics a life-insurance renewal ledger.

Produces ONE table of due events (the grain the model works at):
    policy_id, due_date, policy_year, ... static/slow attributes ...,
    days_to_pay (float, NaN if never paid within the observation window)

Replace `generate()` with a SQL pull of the same shape and the rest of the
pipeline is unchanged, which is why it is isolated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG

MODES = ["auto_debit", "manual_online", "manual_branch", "agent_collected"]
MODE_P = [0.34, 0.28, 0.18, 0.20]
PRODUCTS = ["term", "endowment", "ulip", "money_back", "whole_life"]
CHANNELS = ["agency", "bancassurance", "direct", "broker"]
FREQS = {"yearly": 365, "half_yearly": 182, "quarterly": 91}


def generate(n_policies: int = 20_000, start: str = "2019-01-01",
             end: str = "2025-12-31", seed: int = CFG.seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # --- policy master ------------------------------------------------------
    mode = rng.choice(MODES, n_policies, p=MODE_P)
    product = rng.choice(PRODUCTS, n_policies, p=[0.22, 0.28, 0.18, 0.17, 0.15])
    channel = rng.choice(CHANNELS, n_policies, p=[0.45, 0.25, 0.12, 0.18])
    freq = rng.choice(list(FREQS), n_policies, p=[0.6, 0.22, 0.18])
    annual_premium = np.round(rng.lognormal(9.9, 0.75, n_policies), -2)
    age = rng.integers(21, 66, n_policies)
    sum_assured_mult = rng.choice([10, 15, 20, 25], n_policies)
    inception = start + pd.to_timedelta(
        rng.integers(-6 * 365, int((end - start).days * 0.6), n_policies), unit="D")

    # latent per-policy discipline: the thing the model has to infer from history
    discipline = rng.normal(0, 1, n_policies)
    discipline += np.where(mode == "auto_debit", 1.1, 0.0)
    discipline += np.where(channel == "direct", 0.25, 0.0)
    discipline -= np.where(product == "ulip", 0.35, 0.0)
    discipline -= 0.20 * (np.log(annual_premium) - np.log(annual_premium).mean())

    agent_active = rng.random(n_policies) < 0.72

    rows = []
    for i in range(n_policies):
        step = FREQS[freq[i]]
        due = inception[i] + pd.Timedelta(days=step)
        pol_year = 1
        prior_lapses = 0
        hist: list[float] = []           # days-late on prior dues (grace-capped)
        while due <= end:
            # policy-year effect: lapses cluster in years 1-3, then settle
            year_pen = {1: 0.55, 2: 0.35, 3: 0.20}.get(pol_year, 0.0)
            # mild economic drift by calendar year
            drift = 0.12 * (due.year - 2019) / 6.0
            z = (discipline[i] - year_pen - drift
                 - 0.30 * min(prior_lapses, 3)
                 - (0.0 if agent_active[i] else 0.25)
                 + 0.20 * (np.mean(hist[-3:]) < 5 if hist else 0))
            p_pay = 1 / (1 + np.exp(-(3.10 + 1.05 * z)))
            if rng.random() < p_pay:
                # spiky, bimodal: on-time cluster near 0, procrastinators near grace
                if rng.random() < 1 / (1 + np.exp(-(0.6 + z))):
                    d = float(min(CFG.grace_days, rng.gamma(1.4, 3.0)))
                else:
                    d = float(min(CFG.grace_days,
                                  CFG.grace_days - abs(rng.normal(0, 8))))
                days_to_pay = round(d)
            else:
                # lapsed at grace; some revive later (drives the hazard model)
                if rng.random() < 0.28:
                    days_to_pay = float(CFG.grace_days + rng.integers(5, 320))
                else:
                    days_to_pay = np.nan
                prior_lapses += 1

            rows.append((f"P{i:07d}", due, pol_year, freq[i], mode[i], product[i],
                         channel[i], float(annual_premium[i]), int(age[i]),
                         int(sum_assured_mult[i]), bool(agent_active[i]),
                         days_to_pay))
            hist.append(min(days_to_pay, CFG.grace_days) if days_to_pay == days_to_pay
                        else CFG.grace_days)
            due += pd.Timedelta(days=step)
            pol_year = int((due - inception[i]).days // 365) + 1

    df = pd.DataFrame(rows, columns=[
        "policy_id", "due_date", "policy_year", "premium_freq", "payment_mode",
        "product", "channel", "annual_premium", "cust_age", "sum_assured_mult",
        "agent_active", "days_to_pay"])
    return df.sort_values(["policy_id", "due_date"], ignore_index=True)
