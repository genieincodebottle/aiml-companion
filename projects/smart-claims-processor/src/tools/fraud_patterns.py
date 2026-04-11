"""
Fraud Pattern Database Tools - used by the CrewAI Fraud Detection Crew.

Contains:
- Known fraud patterns (rule-based first pass before LLM)
- Statistical baseline data for anomaly detection
- Helper functions for each crew member
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PATTERNS_FILE = Path("./data/fraud_patterns/known_patterns.json")

# ── Known Fraud Patterns (rule-based fast check) ──────────────────────────────

KNOWN_PATTERNS = [
    {
        "id": "FP-001",
        "name": "New Policy Short Claim",
        "description": "Claim filed within 30 days of policy start - common fraud indicator",
        "risk_weight": 0.6,
        "check": lambda claim, policy: _days_after_start(claim, policy) <= 30,
    },
    {
        "id": "FP-002",
        "name": "Round Number Amount",
        "description": "Estimated amount is exactly a round number (staged claim indicator)",
        "risk_weight": 0.3,
        "check": lambda claim, policy: float(claim.get("estimated_amount", 0)) % 1000 == 0,
    },
    {
        "id": "FP-003",
        "name": "Weekend/Holiday Incident",
        "description": "High proportion of fraudulent claims occur on weekends with no witnesses",
        "risk_weight": 0.2,
        "check": lambda claim, policy: _is_weekend(claim.get("incident_date", "")),
    },
    {
        "id": "FP-004",
        "name": "No Police Report Auto",
        "description": "Auto collision claim without police report number",
        "risk_weight": 0.4,
        "check": lambda claim, policy: (
            claim.get("incident_type", "") == "auto_collision"
            and not claim.get("police_report_number")
        ),
    },
    {
        "id": "FP-005",
        "name": "Repeat High-Value Claims",
        "description": "Multiple large claims from same claimant",
        "risk_weight": 0.7,
        "check": lambda claim, policy: policy.get("claims_count", 0) >= 2,
    },
    {
        "id": "FP-006",
        "name": "Amount Exceeds Vehicle Value",
        "description": "Claimed damage exceeds typical market value of vehicle year",
        "risk_weight": 0.65,
        "check": lambda claim, policy: _amount_exceeds_vehicle_value(claim),
    },
]

# ── Statistical Baselines (for Anomaly Detector) ──────────────────────────────

CLAIM_BASELINES = {
    "auto_collision": {
        "avg_amount": 6500,
        "std_dev": 3200,
        "median_amount": 5800,
        "p95_amount": 18000,
    },
    "auto_theft": {
        "avg_amount": 22000,
        "std_dev": 12000,
        "median_amount": 18500,
        "p95_amount": 55000,
    },
    "property_fire": {
        "avg_amount": 45000,
        "std_dev": 28000,
        "median_amount": 32000,
        "p95_amount": 120000,
    },
    "property_water": {
        "avg_amount": 12000,
        "std_dev": 8000,
        "median_amount": 9500,
        "p95_amount": 35000,
    },
    "liability": {
        "avg_amount": 18000,
        "std_dev": 14000,
        "median_amount": 12000,
        "p95_amount": 75000,
    },
    "medical": {
        "avg_amount": 8500,
        "std_dev": 5500,
        "median_amount": 6200,
        "p95_amount": 28000,
    },
}


# ── Helper Functions ──────────────────────────────────────────────────────────

def _days_after_start(claim: dict, policy: dict) -> int:
    from datetime import date
    try:
        start = date.fromisoformat(policy.get("start_date", "2000-01-01"))
        incident = date.fromisoformat(claim.get("incident_date", "2000-01-01"))
        return max(0, (incident - start).days)
    except ValueError:
        return 999


def _is_weekend(date_str: str) -> bool:
    from datetime import date
    try:
        d = date.fromisoformat(date_str)
        return d.weekday() >= 5  # 5=Saturday, 6=Sunday
    except ValueError:
        return False


def _amount_exceeds_vehicle_value(claim: dict) -> bool:
    """Simple vehicle value check using year + typical depreciation."""
    from datetime import date
    year = claim.get("vehicle_year")
    amount = float(claim.get("estimated_amount", 0))
    if not year:
        return False
    age = date.today().year - int(year)
    # Rough baseline: new car $35K, depreciates ~15%/yr
    est_value = max(3000, 35000 * (0.85 ** age))
    return amount > est_value * 1.2  # Flag if >120% of estimated value


# ── Public API ────────────────────────────────────────────────────────────────

def check_known_patterns(claim: dict, policy: dict) -> tuple[list[str], float]:
    """
    Run all rule-based fraud patterns.
    Returns (matched_pattern_names, composite_risk_score).
    """
    matched = []
    total_weight = 0.0
    max_possible = sum(p["risk_weight"] for p in KNOWN_PATTERNS)

    for pattern in KNOWN_PATTERNS:
        try:
            if pattern["check"](claim, policy):
                matched.append(f"{pattern['id']}: {pattern['name']} - {pattern['description']}")
                total_weight += pattern["risk_weight"]
        except Exception:
            continue

    risk_score = min(total_weight / max_possible if max_possible > 0 else 0, 1.0)
    return matched, risk_score


def get_statistical_anomaly(claim_type: str, amount: float) -> dict:
    """
    Check if a claim amount is statistically anomalous for its type.
    Returns z-score and anomaly assessment.
    """
    baseline = CLAIM_BASELINES.get(claim_type, CLAIM_BASELINES["auto_collision"])
    avg = baseline["avg_amount"]
    std = baseline["std_dev"]
    z_score = (amount - avg) / std if std > 0 else 0

    return {
        "z_score": round(z_score, 2),
        "average_for_type": avg,
        "claim_amount": amount,
        "is_outlier": abs(z_score) > 2.0,
        "is_extreme_outlier": abs(z_score) > 3.0,
        "percentile_estimate": "95th+" if amount > baseline["p95_amount"] else "normal range",
    }
