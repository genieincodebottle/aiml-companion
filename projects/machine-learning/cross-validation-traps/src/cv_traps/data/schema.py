"""The ingest contract. An upstream change should fail here, loudly.

Swap the generator for a real warehouse query and this is the file that tells
you the export changed shape, instead of the model quietly getting worse.
"""
from __future__ import annotations

import pandas as pd

from cv_traps.data.generate import (CUSTOMER_FEATURES, ID_COLUMNS,
                                    PERIOD_FEATURES, TARGET)


class SchemaError(ValueError):
    """Raised when the incoming frame breaks the contract."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = list(ID_COLUMNS) + list(CUSTOMER_FEATURES) + \
        list(PERIOD_FEATURES) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    if df["row_id"].duplicated().any():
        raise SchemaError("row_id is not unique, so the grain is wrong")

    if df.duplicated(["customer_id", "period"]).any():
        raise SchemaError(
            "a customer appears twice in one period; the grain is supposed to "
            "be one row per customer-period, and grouped folds assume it")

    if not set(df[TARGET].unique()) <= {0, 1}:
        raise SchemaError(f"{TARGET} must be binary")

    rate = float(df[TARGET].mean())
    if not 0.02 < rate < 0.60:
        raise SchemaError(
            f"churn rate is {rate:.1%}, outside the range this project's "
            "conclusions were measured at; the panel is probably misconfigured")

    # The customer-level columns must actually be constant within a customer,
    # because the group trap depends on them fingerprinting the customer.
    for col in CUSTOMER_FEATURES:
        if df.groupby("customer_id")[col].nunique().max() > 1:
            raise SchemaError(
                f"{col} varies within a customer; it is documented as a "
                "customer-level attribute and grouped validation assumes so")

    if df["period"].nunique() < 4:
        raise SchemaError(
            "fewer than 4 periods, so forward-chaining validation cannot run")
    return df


def summarise(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "customers": int(df["customer_id"].nunique()),
        "periods": int(df["period"].nunique()),
        "features": int(len([c for c in df.columns
                             if c not in set(ID_COLUMNS) | {TARGET}])),
        "churn_rate": round(float(df[TARGET].mean()), 4),
        "rows_per_customer": round(
            float(df.groupby("customer_id").size().mean()), 2),
        "pct_missing": round(float(df.isna().to_numpy().mean()), 4),
    }
