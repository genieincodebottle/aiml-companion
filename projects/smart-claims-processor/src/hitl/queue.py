"""
HITL (Human-In-The-Loop) Review Queue with FastAPI endpoints.

Architecture:
  - SQLite-backed queue (no extra services required)
  - FastAPI router mounted at /hitl
  - Priority ordering: CRITICAL > HIGH > NORMAL
  - SLA tracking: alerts if ticket exceeds SLA hours
  - Human review recorded in audit log

Endpoints:
  POST /hitl/enqueue              - Add claim to review queue
  GET  /hitl/queue                - List pending reviews (by priority)
  GET  /hitl/ticket/{ticket_id}   - Get full review brief
  POST /hitl/decide/{ticket_id}   - Submit human decision
  GET  /hitl/stats                - Queue statistics
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import get_hitl_config
from src.models.schemas import ClaimDecision, HITLPriority
from src.security.audit_log import log_hitl_event

logger = logging.getLogger(__name__)

DB_PATH = Path("./data/hitl_queue.db")
router = APIRouter(prefix="/hitl", tags=["HITL Review Queue"])

# ── Priority ordering for queue ───────────────────────────────────────────────
_PRIORITY_ORDER = {
    HITLPriority.CRITICAL.value: 0,
    HITLPriority.HIGH.value: 1,
    HITLPriority.NORMAL.value: 2,
}


# ── Database Setup ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hitl_queue (
            ticket_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            priority TEXT NOT NULL,
            priority_score REAL NOT NULL,
            triggers TEXT NOT NULL,
            review_brief TEXT NOT NULL,
            state_snapshot TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            sla_deadline TEXT NOT NULL,
            resolved_at TEXT,
            reviewer_id TEXT,
            human_decision TEXT,
            human_notes TEXT,
            override_ai INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


# ── Pydantic Models ───────────────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    claim_id: str
    priority: str
    priority_score: float
    triggers: list[str]
    review_brief: str
    state_snapshot: dict


class DecisionRequest(BaseModel):
    reviewer_id: str
    decision: str                  # One of ClaimDecision values
    settlement_override_usd: Optional[float] = None
    notes: str = ""
    override_ai: bool = False


class TicketSummary(BaseModel):
    ticket_id: str
    claim_id: str
    priority: str
    priority_score: float
    status: str
    created_at: str
    sla_deadline: str
    triggers: list[str]


# ── Queue Operations ──────────────────────────────────────────────────────────

def enqueue_claim(
    claim_id: str,
    priority: HITLPriority,
    priority_score: float,
    triggers: list[str],
    review_brief: str,
    state_snapshot: dict,
) -> str:
    """Add a claim to the HITL review queue. Returns ticket_id."""
    cfg = get_hitl_config()
    sla_hours = cfg["sla_hours"].get(priority.value, 72)

    ticket_id = f"HITL-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    sla_deadline = now + timedelta(hours=sla_hours)

    conn = _get_db()
    try:
        conn.execute("""
            INSERT INTO hitl_queue
            (ticket_id, claim_id, priority, priority_score, triggers, review_brief,
             state_snapshot, status, created_at, sla_deadline)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """, (
            ticket_id,
            claim_id,
            priority.value,
            priority_score,
            json.dumps(triggers),
            review_brief,
            json.dumps(state_snapshot, default=str),
            now.isoformat(),
            sla_deadline.isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()

    log_hitl_event(
        claim_id=claim_id,
        event="ENQUEUED",
        priority=priority.value,
        triggers=triggers,
    )

    logger.info(f"HITL ticket {ticket_id} created for claim {claim_id} | Priority: {priority.value}")
    return ticket_id


def get_human_decision(ticket_id: str, timeout_seconds: int = 30) -> Optional[dict]:
    """
    Poll the queue for a human decision on a ticket.
    Returns the decision dict if resolved, None if still pending.
    Used by the LangGraph interrupt/resume pattern.
    """
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None
    if row["status"] != "resolved":
        return None

    return {
        "decision": row["human_decision"],
        "reviewer_id": row["reviewer_id"],
        "notes": row["human_notes"],
        "override_ai": bool(row["override_ai"]),
        "resolved_at": row["resolved_at"],
    }


# ── FastAPI Endpoints ─────────────────────────────────────────────────────────

@router.get("/queue", response_model=list[TicketSummary])
def list_pending_reviews(status: str = "pending"):
    """List all tickets sorted by priority then created_at."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM hitl_queue WHERE status = ? ORDER BY priority_score DESC, created_at ASC",
            (status,)
        ).fetchall()
    finally:
        conn.close()

    return [
        TicketSummary(
            ticket_id=r["ticket_id"],
            claim_id=r["claim_id"],
            priority=r["priority"],
            priority_score=r["priority_score"],
            status=r["status"],
            created_at=r["created_at"],
            sla_deadline=r["sla_deadline"],
            triggers=json.loads(r["triggers"]),
        )
        for r in rows
    ]


@router.get("/ticket/{ticket_id}")
def get_ticket(ticket_id: str):
    """Get full ticket details including review brief and state snapshot."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    return {
        "ticket_id": row["ticket_id"],
        "claim_id": row["claim_id"],
        "priority": row["priority"],
        "priority_score": row["priority_score"],
        "status": row["status"],
        "review_brief": row["review_brief"],
        "triggers": json.loads(row["triggers"]),
        "created_at": row["created_at"],
        "sla_deadline": row["sla_deadline"],
        "resolved_at": row["resolved_at"],
        "reviewer_id": row["reviewer_id"],
        "human_decision": row["human_decision"],
        "human_notes": row["human_notes"],
    }


@router.post("/decide/{ticket_id}")
def submit_decision(ticket_id: str, body: DecisionRequest):
    """Submit a human decision for a ticket. Resolves the ticket."""
    # Validate decision value
    valid_decisions = [d.value for d in ClaimDecision]
    if body.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{body.decision}'. Valid: {valid_decisions}"
        )

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT claim_id FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

        claim_id = row["claim_id"]
        now = datetime.now(timezone.utc).isoformat()

        conn.execute("""
            UPDATE hitl_queue
            SET status = 'resolved', resolved_at = ?, reviewer_id = ?,
                human_decision = ?, human_notes = ?, override_ai = ?
            WHERE ticket_id = ?
        """, (
            now,
            body.reviewer_id,
            body.decision,
            body.notes,
            1 if body.override_ai else 0,
            ticket_id,
        ))
        conn.commit()
    finally:
        conn.close()

    log_hitl_event(
        claim_id=claim_id,
        event="RESOLVED",
        priority="",
        triggers=[],
        reviewer_id=body.reviewer_id,
        human_decision=body.decision,
        human_notes=body.notes,
        override_ai=body.override_ai,
    )

    logger.info(f"HITL ticket {ticket_id} resolved: {body.decision} by {body.reviewer_id}")
    return {"status": "resolved", "ticket_id": ticket_id, "decision": body.decision}


@router.get("/stats")
def queue_stats():
    """Summary statistics for the HITL queue."""
    conn = _get_db()
    try:
        pending = conn.execute("SELECT COUNT(*) FROM hitl_queue WHERE status = 'pending'").fetchone()[0]
        resolved_today = conn.execute(
            "SELECT COUNT(*) FROM hitl_queue WHERE status = 'resolved' AND DATE(resolved_at) = DATE('now')"
        ).fetchone()[0]
        critical = conn.execute(
            "SELECT COUNT(*) FROM hitl_queue WHERE status = 'pending' AND priority = 'critical'"
        ).fetchone()[0]
        high = conn.execute(
            "SELECT COUNT(*) FROM hitl_queue WHERE status = 'pending' AND priority = 'high'"
        ).fetchone()[0]
        overrides = conn.execute(
            "SELECT COUNT(*) FROM hitl_queue WHERE override_ai = 1 AND DATE(resolved_at) = DATE('now')"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "pending_total": pending,
        "pending_critical": critical,
        "pending_high": high,
        "resolved_today": resolved_today,
        "human_overrides_today": overrides,
    }
