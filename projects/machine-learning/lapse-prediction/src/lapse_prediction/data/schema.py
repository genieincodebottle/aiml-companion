"""The contract between whatever produces the renewal ledger and this package.

In production the ledger comes from a warehouse query that someone else owns
and will eventually change. Validating the contract at the boundary turns a
silent model degradation into a loud failure at ingest, which is the whole
point of having one.
"""
from __future__ import annotations

import pandas as pd

# column -> (required, pandas dtype kind check)
REQUIRED: dict[str, str] = {
    "policy_id": "object",
    "due_date": "datetime",
    "policy_year": "numeric",
    "premium_freq": "object",
    "payment_mode": "object",
    "product": "object",
    "channel": "object",
    "annual_premium": "numeric",
    "cust_age": "numeric",
    "sum_assured_mult": "numeric",
    "agent_active": "any",
    "days_to_pay": "numeric",   # NaN when the premium was never received
}


class SchemaError(ValueError):
    pass


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "datetime":
        return pd.api.types.is_datetime64_any_dtype(s)
    if kind == "numeric":
        return pd.api.types.is_numeric_dtype(s)
    return not pd.api.types.is_numeric_dtype(s)   # "object"-ish


def validate(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Check the ledger. Extra columns are allowed and flow through to features."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"ledger is missing required columns: {missing}")

    bad = [f"{c} (expected {k}, got {df[c].dtype})"
           for c, k in REQUIRED.items() if not _kind_ok(df[c], k)]
    if bad and strict:
        raise SchemaError("column type mismatch: " + "; ".join(bad))

    problems = []
    if df["policy_id"].isna().any():
        problems.append("policy_id contains nulls")
    if df["due_date"].isna().any():
        problems.append("due_date contains nulls")
    if df.duplicated(["policy_id", "due_date"]).any():
        n = int(df.duplicated(["policy_id", "due_date"]).sum())
        problems.append(f"{n} duplicate (policy_id, due_date) rows -- the grain is broken")
    neg = df["days_to_pay"].dropna() < 0
    if neg.any():
        problems.append(f"{int(neg.sum())} rows with negative days_to_pay")
    if (df["annual_premium"].dropna() <= 0).any():
        problems.append("non-positive annual_premium")

    if problems and strict:
        raise SchemaError("; ".join(problems))
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Row counts and rates worth logging on every ingest, so drift is visible."""
    d = df["days_to_pay"]
    return {
        "rows": int(len(df)),
        "policies": int(df["policy_id"].nunique()),
        "date_min": str(df["due_date"].min().date()),
        "date_max": str(df["due_date"].max().date()),
        "never_paid_rate": round(float(d.isna().mean()), 4),
        "median_days_to_pay": float(d.median()),
        "null_rate_max": round(float(df.isna().mean().max()), 4),
    }
