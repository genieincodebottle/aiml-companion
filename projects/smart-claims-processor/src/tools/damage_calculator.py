"""
Damage Calculator Tools - used by the Damage Assessment Agent.

Provides depreciation tables, repair cost baselines,
and total-loss threshold calculations.
"""

from __future__ import annotations

from datetime import date
from typing import Optional


# Depreciation rates by asset type and age
DEPRECIATION_RATES = {
    "auto": {
        "age_0_1": 0.20,
        "age_1_2": 0.15,
        "age_2_3": 0.12,
        "age_3_5": 0.09,
        "age_5_plus": 0.07,
    },
    "property": {
        "age_0_5": 0.04,
        "age_5_10": 0.06,
        "age_10_20": 0.08,
        "age_20_plus": 0.10,
    },
}

# Auto repair cost ranges by component (min, avg, max)
# USD baselines (US)
_REPAIR_BASELINES_USD = {
    "trunk_damage": (800, 2200, 5500),
    "front_bumper": (600, 1800, 4000),
    "rear_bumper": (500, 1600, 3500),
    "hood": (700, 2000, 5000),
    "door_panel": (400, 1200, 3000),
    "windshield": (300, 650, 1200),
    "fender": (400, 1100, 2800),
    "frame_damage": (1500, 4500, 12000),
    "engine_damage": (2000, 6000, 20000),
    "total_interior": (1000, 3500, 8000),
}
# INR baselines (India) - typical authorized garage rates
_REPAIR_BASELINES_INR = {
    "trunk_damage": (8000, 20000, 55000),
    "front_bumper": (5000, 15000, 40000),
    "rear_bumper": (4000, 12000, 35000),
    "hood": (6000, 18000, 50000),
    "door_panel": (4000, 10000, 30000),
    "windshield": (3000, 8000, 15000),
    "fender": (3500, 10000, 25000),
    "frame_damage": (15000, 45000, 120000),
    "engine_damage": (20000, 60000, 200000),
    "total_interior": (10000, 35000, 80000),
}

def _get_repair_baselines() -> dict:
    """Return repair cost baselines for the active country."""
    try:
        from src.config import get_country_meta
        code = get_country_meta().get("code", "US")
        if code == "IN":
            return _REPAIR_BASELINES_INR
    except Exception:
        pass
    return _REPAIR_BASELINES_USD

# DEPRECATED: Use _get_repair_baselines() instead, which is country-aware.
# This constant is kept for backward compatibility only.
AUTO_REPAIR_BASELINES = _REPAIR_BASELINES_USD

# Total loss threshold: if repair cost > X% of ACV, declare total loss
TOTAL_LOSS_THRESHOLD = 0.75


def calculate_vehicle_acv(
    year: int,
    make: str,
    model: str,
    mileage: Optional[int] = None,
) -> float:
    """
    Estimate Actual Cash Value (ACV) of a vehicle.
    Uses simplified model - production would use NADA/KBB API.
    """
    age = date.today().year - year

    # Base vehicle value by age - country-aware
    try:
        from src.config import get_country_meta
        is_india = get_country_meta().get("code", "US") == "IN"
    except Exception:
        is_india = False

    if is_india:
        # Indian market values (INR) - typical mid-segment car
        if age <= 1:
            base = 800000
        elif age <= 3:
            base = 600000
        elif age <= 6:
            base = 400000
        elif age <= 10:
            base = 250000
        else:
            base = 120000
    else:
        # US market values (USD)
        if age <= 1:
            base = 35000
        elif age <= 3:
            base = 28000
        elif age <= 6:
            base = 20000
        elif age <= 10:
            base = 13000
        else:
            base = 7000

    # Apply cumulative depreciation
    rates = DEPRECIATION_RATES["auto"]
    acv = base
    remaining_age = age
    deductions = {
        1: rates["age_0_1"],
        2: rates["age_1_2"],
        3: rates["age_2_3"],
        5: rates["age_3_5"],
    }
    for threshold, rate in sorted(deductions.items()):
        if remaining_age > 0:
            years_in_bracket = min(remaining_age, threshold)
            acv *= (1 - rate) ** years_in_bracket
            remaining_age -= years_in_bracket

    # Mileage adjustment
    if mileage:
        expected_mileage = age * 12000
        if mileage > expected_mileage * 1.5:
            acv *= 0.90  # High mileage penalty
        elif mileage < expected_mileage * 0.5:
            acv *= 1.08  # Low mileage premium

    return round(acv, 2)


def should_total_loss(repair_estimate: float, acv: float) -> tuple[bool, float]:
    """
    Determine if a vehicle should be declared a total loss.
    Returns (is_total_loss, repair_to_acv_ratio).
    """
    if acv <= 0:
        return False, 0.0
    ratio = repair_estimate / acv
    return ratio >= TOTAL_LOSS_THRESHOLD, round(ratio, 3)


def apply_depreciation(
    damage_amount: float,
    asset_type: str,
    asset_age_years: int,
) -> tuple[float, float]:
    """
    Apply depreciation to a damage amount.
    Returns (depreciated_amount, depreciation_applied).
    """
    rates = DEPRECIATION_RATES.get(asset_type, DEPRECIATION_RATES["auto"])

    if asset_type == "auto":
        if asset_age_years <= 1:
            rate = rates["age_0_1"]
        elif asset_age_years <= 2:
            rate = rates["age_1_2"]
        elif asset_age_years <= 3:
            rate = rates["age_2_3"]
        elif asset_age_years <= 5:
            rate = rates["age_3_5"]
        else:
            rate = rates["age_5_plus"]
    else:  # property
        if asset_age_years <= 5:
            rate = rates["age_0_5"]
        elif asset_age_years <= 10:
            rate = rates["age_5_10"]
        elif asset_age_years <= 20:
            rate = rates["age_10_20"]
        else:
            rate = rates["age_20_plus"]

    depreciation = damage_amount * rate * asset_age_years
    depreciation = min(depreciation, damage_amount * 0.70)  # Max 70% depreciation
    depreciated = max(0, damage_amount - depreciation)

    return round(depreciated, 2), round(depreciation, 2)


def apply_depreciation_country_aware(
    damage_amount: float,
    asset_type: str,
    asset_age_years: int,
) -> tuple[float, float, str]:
    """
    Country-aware depreciation. Returns (depreciated_amount, depreciation_applied, method_used).

    US: year-based (existing logic above).
    India: part-wise IRDAI rates. Since we don't have an itemized surveyor's
    parts list, we apply a blended average across typical part categories.
    In production a surveyor's report would provide exact part breakdowns.
    """
    try:
        from src.config import get_depreciation_config
        dep_cfg = get_depreciation_config()
    except Exception:
        dep_cfg = {}

    method = dep_cfg.get("method", "year_based")

    if method == "part_wise":
        parts = dep_cfg.get("parts", {})
        if parts:
            # Typical auto claim part mix weights (approximate)
            _PART_WEIGHTS = {
                "metal_body": 0.35,
                "paint": 0.15,
                "rubber_nylon_plastic": 0.15,
                "glass": 0.10,
                "electrical_parts": 0.10,
                "mechanical_parts": 0.10,
                "batteries": 0.03,
                "air_bags": 0.02,
            }
            blended_rate = sum(
                parts.get(part, 0.0) * weight
                for part, weight in _PART_WEIGHTS.items()
            )
        else:
            blended_rate = 0.15
        depreciation = damage_amount * blended_rate
        depreciation = min(depreciation, damage_amount * 0.70)
        depreciated = max(0, damage_amount - depreciation)
        return round(depreciated, 2), round(depreciation, 2), "part_wise (IRDAI blended)"

    # Default: year-based (US)
    depreciated, dep_applied = apply_depreciation(damage_amount, asset_type, asset_age_years)
    return depreciated, dep_applied, "year_based"


def get_repair_estimate_range(damage_description: str) -> Optional[tuple[float, float, float]]:
    """
    Look up repair cost range for common damage types.
    Returns (min, avg, max) or None.
    """
    baselines = _get_repair_baselines()
    desc_lower = damage_description.lower()
    for keyword, ranges in baselines.items():
        if any(word in desc_lower for word in keyword.split("_")):
            return ranges
    return None
