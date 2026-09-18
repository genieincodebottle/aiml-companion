"""
HITL (Human-In-The-Loop) Review Queue - storage and queue operations.

Architecture:
  - SQLite-backed queue (no extra services required)
  - HTTP endpoints live in api/routes_hitl.py (mounted at /api/hitl)
  - Priority ordering: CRITICAL > HIGH > NORMAL
  - SLA tracking: a pending ticket past its deadline is escalated to CRITICAL
    and the escalation is written to the audit log
  - Idempotent enqueue: LangGraph re-runs a node from its first line when the
    graph resumes after interrupt(), so the node that creates a ticket runs
    twice for one pause. The idempotency key makes the second call return the
    ticket the first call created instead of opening a duplicate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import get_hitl_config
from src.models.schemas import HITLPriority
from src.security.audit_log import log_hitl_event

logger = logging.getLogger(__name__)

DB_PATH = Path("./data/hitl_queue.db")

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
    # Columns added after the first release: migrate existing databases in place.
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(hitl_queue)")}
    if "idempotency_key" not in existing:
        conn.execute("ALTER TABLE hitl_queue ADD COLUMN idempotency_key TEXT")
    if "sla_breached" not in existing:
        conn.execute("ALTER TABLE hitl_queue ADD COLUMN sla_breached INTEGER DEFAULT 0")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_hitl_idempotency "
        "ON hitl_queue(idempotency_key) WHERE idempotency_key IS NOT NULL"
    )
    conn.commit()
    return conn


# ── Queue Operations ──────────────────────────────────────────────────────────

def enqueue_claim(
    claim_id: str,
    priority: HITLPriority,
    priority_score: float,
    triggers: list[str],
    review_brief: str,
    state_snapshot: dict,
    idempotency_key: Optional[str] = None,
) -> str:
    """Add a claim to the HITL review queue. Returns ticket_id.

    With an idempotency_key, calling this again for the same pause returns the
    existing ticket and writes nothing.
    """
    conn = _get_db()
    try:
        if idempotency_key:
            row = conn.execute(
                "SELECT ticket_id FROM hitl_queue WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row:
                logger.info(f"HITL ticket {row['ticket_id']} already exists for {idempotency_key}; reusing")
                return row["ticket_id"]

        cfg = get_hitl_config()
        sla_hours = cfg["sla_hours"].get(priority.value, 72)
        ticket_id = f"HITL-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)
        sla_deadline = now + timedelta(hours=sla_hours)

        conn.execute("""
            INSERT INTO hitl_queue
            (ticket_id, claim_id, priority, priority_score, triggers, review_brief,
             state_snapshot, status, created_at, sla_deadline, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
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
            idempotency_key,
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


def escalate_overdue_tickets(conn: sqlite3.Connection, now: Optional[datetime] = None) -> list[str]:
    """Escalate pending tickets past their SLA deadline to CRITICAL.

    Called whenever the queue is read, so a reviewer opening the queue always
    sees overdue work at the top. Each escalation happens once per ticket and
    is written to the audit log. Returns the escalated ticket ids.
    """
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT ticket_id, claim_id, priority, triggers, sla_deadline FROM hitl_queue "
        "WHERE status = 'pending' AND COALESCE(sla_breached, 0) = 0 AND sla_deadline < ?",
        (now.isoformat(),),
    ).fetchall()
    escalated = []
    for row in rows:
        conn.execute(
            "UPDATE hitl_queue SET sla_breached = 1, priority = ?, "
            "priority_score = MAX(priority_score, 80) WHERE ticket_id = ?",
            (HITLPriority.CRITICAL.value, row["ticket_id"]),
        )
        log_hitl_event(
            claim_id=row["claim_id"],
            event="SLA_BREACHED_ESCALATED",
            priority=HITLPriority.CRITICAL.value,
            triggers=[f"SLA deadline {row['sla_deadline']} passed while pending "
                      f"(was {row['priority']})"],
        )
        escalated.append(row["ticket_id"])
    if escalated:
        conn.commit()
        logger.warning(f"Escalated {len(escalated)} overdue HITL ticket(s) to critical: {escalated}")
    return escalated


def get_human_decision(ticket_id: str) -> Optional[dict]:
    """Return the recorded human decision for a ticket, or None if still pending."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row or row["status"] != "resolved":
        return None

    return {
        "decision": row["human_decision"],
        "reviewer_id": row["reviewer_id"],
        "notes": row["human_notes"],
        "override_ai": bool(row["override_ai"]),
        "resolved_at": row["resolved_at"],
    }
