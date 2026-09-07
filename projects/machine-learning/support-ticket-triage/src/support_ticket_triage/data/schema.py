"""The ingest contract. An upstream change should fail here, loudly.

Swap the generator for a real ticket export and this is the file that tells you
the export changed shape, instead of the model quietly getting worse.
"""
from __future__ import annotations

import pandas as pd

from support_ticket_triage.data.generate import CLASSES, VOCAB

REQUIRED = ("ticket_id", "category", "text", "n_tokens")


class SchemaError(ValueError):
    """Raised when the incoming frame breaks the contract."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    absent_tokens = [t for t in VOCAB if t not in df.columns]
    if absent_tokens:
        raise SchemaError(
            f"{len(absent_tokens)} vocabulary columns are missing, starting with "
            f"{absent_tokens[:5]}. The feature matrix would silently change width.")

    if df["ticket_id"].duplicated().any():
        raise SchemaError("ticket_id is not unique, so the grain is wrong")

    unknown = set(df["category"].unique()) - set(CLASSES)
    if unknown:
        raise SchemaError(
            f"unknown categories {sorted(unknown)}. A new label class needs a "
            "deliberate decision, not an automatic one.")

    token_block = df[list(VOCAB)]
    if not token_block.isin((0, 1)).all().all():
        raise SchemaError("token columns must be binary presence flags")
    if token_block.isna().any().any():
        raise SchemaError("token columns contain nulls")

    empty = int((token_block.sum(axis=1) == 0).sum())
    if empty > len(df) * 0.02:
        raise SchemaError(
            f"{empty} tickets have no tokens at all ({empty / len(df):.1%}); "
            "the vocabulary probably does not match this export")
    return df


def summarise(df: pd.DataFrame) -> dict:
    counts = df["category"].value_counts(normalize=True)
    return {
        "rows": len(df),
        "vocab": len(VOCAB),
        "classes": int(df["category"].nunique()),
        "rarest_class": str(counts.idxmin()),
        "rarest_share": round(float(counts.min()), 4),
        "median_tokens": int(df["n_tokens"].median()),
    }
