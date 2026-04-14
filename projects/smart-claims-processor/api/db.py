"""SQLite persistence layer via SQLModel.

This module defines the database models and initializes 
the SQLite database for the insurance claim processing API."""

from __future__ import annotations #For Python 3.10+ to allow forward references in type hints without quotes Ensure this is at the top of the file before any imports or code.

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "api.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "user"  # user | reviewer | admin
    created_at: str = Field(default_factory=_utcnow)


class Claim(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    claim_id: str = Field(index=True, unique=True)
    user_id: Optional[int] = Field(default=None, index=True)
    policy_number: str
    claimant_name: str = ""
    claimant_email: str = ""
    claimant_phone: str = ""
    claimant_dob: str = ""
    incident_date: str = ""
    incident_type: str = ""
    incident_description: str = ""
    incident_location: str = ""
    police_report_number: Optional[str] = None
    estimated_amount: float = 0.0
    vehicle_year: Optional[int] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    documents: str = "[]"  # JSON list

    status: str = "submitted"  # submitted | processing | completed | failed
    final_decision: Optional[str] = None
    settlement_amount: Optional[float] = None
    fraud_score: Optional[float] = None
    fraud_risk_level: Optional[str] = None
    evaluation_score: Optional[float] = None
    decided_by: Optional[str] = None      # "AI Agent" or reviewer username
    decided_at: Optional[str] = None      # ISO timestamp of final decision
    hitl_required: bool = False
    hitl_ticket_id: Optional[str] = None
    agent_call_count: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    processing_time_sec: float = 0.0
    denial_reasons: str = "[]"  # JSON list
    communication_message: Optional[str] = None
    appeal_instructions: Optional[str] = None
    agent_outputs: Optional[str] = None  # JSON: serialized per-agent Pydantic outputs
    pipeline_path: str = ""  # JSON list of agents called
    error_log: str = "[]"  # JSON list

    created_at: str = Field(default_factory=_utcnow)
    completed_at: Optional[str] = None


class Appeal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    appeal_id: str = Field(index=True, unique=True)
    claim_id: str = Field(index=True)
    user_id: Optional[int] = None
    reason: str = ""
    status: str = "pending"  # pending | approved | denied
    reviewer_id: Optional[int] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow)
    reviewed_at: Optional[str] = None


def _migrate_claim_columns() -> None:
    """Add new columns to existing claim table if missing."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(claim)").fetchall()}
        migrations = [
            ("denial_reasons", "TEXT DEFAULT '[]'"),
            ("communication_message", "TEXT"),
            ("appeal_instructions", "TEXT"),
            ("agent_outputs", "TEXT"),
            ("decided_by", "TEXT"),
            ("decided_at", "TEXT"),
        ]
        for col, typedef in migrations:
            if col not in existing:
                conn.execute(f"ALTER TABLE claim ADD COLUMN {col} {typedef}")
        conn.commit()
    except Exception:
        pass  # table may not exist yet
    finally:
        conn.close()


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _migrate_claim_columns()


def get_session():
    with Session(engine) as session:
        yield session
