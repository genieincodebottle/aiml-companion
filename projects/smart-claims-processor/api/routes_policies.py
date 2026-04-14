"""
Policy Management - CRUD for the policies.db used by the claims pipeline.

GET    /api/policies/        - List all policies (any authenticated user).
POST   /api/policies/        - Create / update a policy (admin only).
DELETE /api/policies/{num}   - Delete a policy (admin only).
"""
from __future__ import annotations # for Python 3.10-3.11 compatibility, allows forward references in type hints without quotes

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.security import get_current_user, require_role
from src.config import get_country_config, get_country_meta

router = APIRouter(prefix="/api/policies", tags=["Policies"])

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "policies.db"


# --- Country-aware default policy templates ---------------------------------

_US_DEFAULTS = [
    {
        "label": "US Auto (Comprehensive)",
        "policy_number": "POL-AUTO-",
        "holder_name": "",
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
        "label": "US Homeowners",
        "policy_number": "POL-HOME-",
        "holder_name": "",
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
]

_INDIA_DEFAULTS = [
    {
        "label": "India Comprehensive (Own Damage + TP)",
        "policy_number": "POL-COMP-IN-",
        "holder_name": "",
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
        "label": "India Third-Party Only",
        "policy_number": "POL-TP-IN-",
        "holder_name": "",
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
]


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            policy_number TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    return conn


# --- Models ---------------------------------------------------------------

class PolicyCreate(BaseModel):
    policy_number: str = Field(..., min_length=1, examples=["POL-AUTO-001"])
    holder_name: str = Field(..., min_length=1)
    type: str = Field(..., examples=["auto", "homeowners"])
    status: str = Field(default="active", examples=["active", "lapsed"])
    start_date: str = Field(..., examples=["2025-01-01"])
    end_date: str = Field(..., examples=["2027-01-01"])
    deductible: float = 0
    coverage: Dict[str, Any] = Field(default_factory=dict)
    exclusions: List[str] = Field(default_factory=list)
    premium_monthly: float = 0
    claims_count: int = 0


# --- Endpoints --------------------------------------------------------─

@router.get("/defaults")
def get_policy_defaults(_=Depends(get_current_user)):
    """Return country-aware default policy templates for quick-create."""

    meta = get_country_meta()
    code = (meta.get("code") or "US").upper()
    templates = _INDIA_DEFAULTS if code == "IN" else _US_DEFAULTS
    return {
        "country": meta,
        "templates": templates,
    }


@router.get("/")
def list_policies(user=Depends(get_current_user)):
    """Return policies from policies.db, filtered by active country and current user."""

    meta = get_country_meta()
    country_code = (meta.get("code") or "US").upper()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT policy_number, data FROM policies ORDER BY policy_number"
        ).fetchall()
    finally:
        conn.close()
    policies = [json.loads(row[1]) for row in rows]
    # Filter: India policies contain "-IN-" in policy number; others are US
    if country_code == "IN":
        policies = [p for p in policies if "-IN-" in p.get("policy_number", "")]
    else:
        policies = [p for p in policies if "-IN-" not in p.get("policy_number", "")]
    # Non-admin users only see their own policies (matched by holder_name)
    if user.role not in ("admin", "reviewer"):
        username = user.username
        policies = [p for p in policies if p.get("holder_name", "").lower() == username.lower()]
    return policies


@router.post("/")
def upsert_policy(body: PolicyCreate, _=Depends(get_current_user)):
    """Create or update a policy (INSERT OR REPLACE)."""

    data = body.model_dump()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO policies (policy_number, data) VALUES (?, ?)",
            (data["policy_number"], json.dumps(data)),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "policy_number": data["policy_number"]}


@router.delete("/{policy_number}")
def delete_policy(policy_number: str, _=Depends(require_role("admin"))):
    """Delete a policy by its policy_number."""
    
    conn = _get_conn()
    try:
        cur = conn.execute(
            "DELETE FROM policies WHERE policy_number = ?", (policy_number,)
        )
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Policy '{policy_number}' not found")
    finally:
        conn.close()
    return {"ok": True, "deleted": policy_number}
