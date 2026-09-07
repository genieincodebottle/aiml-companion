"""
Seed policies.db with test policies for both US and India country profiles.

Run once after cloning:
    python scripts/seed_policies.py

Creates/updates data/policies.db with 4 test policies (2 US, 2 India) that
have valid start/end dates spanning 2025-2027, so the intake agent accepts
them. Also updates the 3 original sample policies to have non-expired dates.

Safe to re-run - uses INSERT OR REPLACE so existing rows are updated.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "policies.db"

POLICIES = [
    # ── US test policies ─────────────────────────────────────────────────────
    {
        "policy_number": "POL-AUTO-TEST-US",
        "holder_name": "Test Claimant",
        "type": "auto",
        "status": "active",
        "start_date": "2025-01-01",
        "end_date": "2027-01-01",
        "deductible": 500,
        "coverage": {
            "collision": 50000,
            "comprehensive": 50000,
            "liability": 100000,
            "medical_payments": 5000,
            "uninsured_motorist": 50000,
            "rental_reimbursement": 1500,
        },
        "exclusions": ["racing", "commercial_use"],
        "premium_monthly": 145,
        "claims_count": 0,
    },
    {
        "policy_number": "POL-HOME-TEST-US",
        "holder_name": "Test Claimant",
        "type": "homeowners",
        "status": "active",
        "start_date": "2025-01-01",
        "end_date": "2027-01-01",
        "deductible": 2500,
        "coverage": {
            "dwelling": 350000,
            "personal_property": 100000,
            "liability": 300000,
            "additional_living": 50000,
        },
        "exclusions": ["flood", "earthquake"],
        "premium_monthly": 185,
        "claims_count": 0,
    },
    # ── India test policies ──────────────────────────────────────────────────
    {
        "policy_number": "POL-COMP-IN-TEST",
        "holder_name": "Test Claimant",
        "type": "auto",
        "status": "active",
        "start_date": "2025-06-01",
        "end_date": "2027-06-01",
        "deductible": 2000,
        "coverage": {
            "comprehensive": 800000,
            "third_party": 1500000,
            "collision": 800000,
            "pa_cover": 1500000,
        },
        "exclusions": ["racing", "drunk_driving", "unlicensed_driver"],
        "premium_monthly": 3500,
        "claims_count": 0,
        "idv": 750000,
    },
    {
        "policy_number": "POL-TP-IN-TEST",
        "holder_name": "Test Claimant",
        "type": "auto",
        "status": "active",
        "start_date": "2025-06-01",
        "end_date": "2027-06-01",
        "deductible": 0,
        "coverage": {
            "third_party": 1500000,
            "pa_cover": 1500000,
        },
        "exclusions": ["drunk_driving"],
        "premium_monthly": 1200,
        "claims_count": 0,
    },
    # ── Update original sample policies with non-expired dates ───────────────
    {
        "policy_number": "POL-AUTO-789456",
        "holder_name": "Jane Smith",
        "type": "auto",
        "status": "active",
        "start_date": "2025-01-15",
        "end_date": "2027-01-15",
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
        "start_date": "2025-03-01",
        "end_date": "2027-03-01",
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
        "status": "active",
        "start_date": "2025-06-01",
        "end_date": "2027-06-01",
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


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            policy_number TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)

    for pol in POLICIES:
        conn.execute(
            "INSERT OR REPLACE INTO policies (policy_number, data) VALUES (?, ?)",
            (pol["policy_number"], json.dumps(pol)),
        )

    conn.commit()
    rows = conn.execute("SELECT policy_number FROM policies ORDER BY policy_number").fetchall()
    conn.close()

    print(f"\nSeeded {len(POLICIES)} policies into {DB_PATH}:\n")
    for r in rows:
        print(f"  {r[0]}")
    print(f"\nUS test policies:    POL-AUTO-TEST-US, POL-HOME-TEST-US")
    print(f"India test policies: POL-COMP-IN-TEST, POL-TP-IN-TEST")
    print(f"Sample policies:     POL-AUTO-789456, POL-HOME-334521, POL-AUTO-112233 (dates updated to 2025-2027)\n")


if __name__ == "__main__":
    main()
