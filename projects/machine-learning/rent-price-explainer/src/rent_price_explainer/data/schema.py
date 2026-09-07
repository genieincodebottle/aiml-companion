"""Ingest contract for the listings table."""
from __future__ import annotations

import pandas as pd

REQUIRED = {
    "listing_id": "object", "builtup_area": "numeric", "carpet_area": "numeric",
    "bedrooms": "numeric", "bathrooms": "numeric", "floor": "numeric",
    "age_years": "numeric", "has_lift": "numeric", "has_parking": "numeric",
    "furnishing": "object", "locality": "object", "metro_km": "numeric",
    "school_rating": "numeric", "crime_index": "numeric",
    "monthly_rent": "numeric",
}


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"listings table is missing columns: {missing}")

    problems = []
    if df["listing_id"].duplicated().any():
        problems.append("duplicate listing_id -- the grain is broken")
    for c in ("monthly_rent", "builtup_area"):
        if (df[c].dropna() <= 0).any():
            problems.append(f"non-positive {c}")
    if df["monthly_rent"].isna().any():
        problems.append("null monthly_rent")
    if (df["carpet_area"] > df["builtup_area"]).any():
        problems.append("carpet_area exceeds builtup_area on some rows")

    if problems and strict:
        raise SchemaError("; ".join(problems))
    return df


def summarise(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "median_rent": float(df["monthly_rent"].median()),
        "rent_p05_p95": [float(df["monthly_rent"].quantile(0.05)),
                         float(df["monthly_rent"].quantile(0.95))],
        "median_area": float(df["builtup_area"].median()),
        "area_rent_corr": round(float(
            df["builtup_area"].corr(df["monthly_rent"])), 3),
        "carpet_builtup_corr": round(float(
            df["carpet_area"].corr(df["builtup_area"])), 4),
    }
