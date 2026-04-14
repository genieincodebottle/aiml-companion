"""
Policy Lookup Tools - simulated policy database.

In production this would connect to a real policy management system.
For learning purposes, we use a SQLite mock with realistic sample data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

DB_PATH = Path("./data/policies.db")

SAMPLE_POLICIES = [
    {
        "policy_number": "POL-AUTO-789456",
        "holder_name": "Jane Smith",
        "type": "auto",
        "status": "active",
        "start_date": "2023-01-15",
        "end_date": "2025-01-15",
        "deductible": 1000,
        "coverage": {
            "collision": 50000,
            "comprehensive": 50000,
            "liability": 100000,
            "medical_payments": 5000,
            "rental_reimbursement": 1500,
        },
        "exclusions": ["racing", "commercial_use", "pre_existing_damage"],
        "premium_monthly": 127,
        "claims_count": 0,
    },
    {
        "policy_number": "POL-HOME-334521",
        "holder_name": "Robert Johnson",
        "type": "homeowners",
        "status": "active",
        "start_date": "2024-03-01",
        "end_date": "2025-03-01",
        "deductible": 2500,
        "coverage": {
            "dwelling": 350000,
            "personal_property": 100000,
            "liability": 300000,
            "additional_living": 50000,
        },
        "exclusions": ["flood", "earthquake", "intentional_damage"],
        "premium_monthly": 185,
        "claims_count": 1,
    },
    {
        "policy_number": "POL-AUTO-112233",
        "holder_name": "Maria Garcia",
        "type": "auto",
        "status": "lapsed",
        "start_date": "2023-06-01",
        "end_date": "2024-06-01",       # Lapsed - for testing denial
        "deductible": 500,
        "coverage": {
            "collision": 30000,
            "comprehensive": 30000,
            "liability": 75000,
        },
        "exclusions": ["racing"],
        "premium_monthly": 89,
        "claims_count": 2,
    },
]


_db_initialized = False


def _ensure_db():
    """Initialize DB and seed sample data (once per process)."""
    global _db_initialized
    if _db_initialized:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            policy_number TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    for p in SAMPLE_POLICIES:
        conn.execute(
            "INSERT OR IGNORE INTO policies (policy_number, data) VALUES (?, ?)",
            (p["policy_number"], json.dumps(p))
        )
    conn.commit()
    conn.close()
    _db_initialized = True


def lookup_policy(policy_number: str) -> Optional[dict]:
    """
    Retrieve full policy details by policy number.
    Returns None if policy not found.
    """
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT data FROM policies WHERE policy_number = ?", (policy_number,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    return json.loads(row[0])


def is_policy_active(policy: dict, incident_date: str) -> tuple[bool, str]:
    """
    Check if a policy was active on the incident date.
    Returns (is_active, reason).
    """
    if policy["status"] != "active":
        return False, f"Policy status is '{policy['status']}' (not active)"

    try:
        incident = date.fromisoformat(incident_date)
        start = date.fromisoformat(policy["start_date"])
        end = date.fromisoformat(policy["end_date"])
        if not (start <= incident <= end):
            return False, f"Incident date {incident_date} outside policy period {policy['start_date']} to {policy['end_date']}"
        return True, "Policy active on incident date"
    except ValueError as e:
        return False, f"Date parsing error: {e}"


def get_coverage_for_claim_type(policy: dict, claim_type: str) -> dict:
    """
    Map claim type to applicable coverage limits.
    Returns dict with covered_amount and applicable exclusions.
    """
    coverage = policy.get("coverage", {})
    exclusions = policy.get("exclusions", [])
    deductible = policy.get("deductible", 0)

    # Country-aware mapping from claim type to coverage key.
    # Falls back to a US default if the config loader isn't available yet.
    try:
        from src.config import get_coverage_mapping
        claim_to_coverage = get_coverage_mapping()
    except Exception:
        claim_to_coverage = {
            "auto_collision": "collision",
            "auto_theft": "comprehensive",
            "property_fire": "dwelling",
            "property_water": "dwelling",
            "liability": "liability",
            "medical": "medical_payments",
        }

    coverage_key = claim_to_coverage.get(claim_type, "")
    covered_amount = coverage.get(coverage_key, 0)

    return {
        "covered": covered_amount > 0,
        "coverage_key": coverage_key,
        "coverage_limit": covered_amount,
        "deductible": deductible,
        "exclusions": exclusions,
        "net_max_payout": max(0, covered_amount - deductible),
    }


def get_claim_history_count(policy_number: str, days: int = 365) -> int:
    """Return number of prior claims for this policy within the window."""
    policy = lookup_policy(policy_number)
    if not policy:
        return 0
    # In production: query claims database. Using stored count for mock.
    return policy.get("claims_count", 0)
