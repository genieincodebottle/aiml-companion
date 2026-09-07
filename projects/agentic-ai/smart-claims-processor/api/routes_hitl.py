"""HITL routes under /api/hitl - wraps the existing queue in src/hitl/queue.py."""
from __future__ import annotations # for type hinting within the same file

import json
import logging
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from api.security import get_current_user, require_role
from src.hitl.queue import _get_db

router = APIRouter(prefix="/api/hitl", tags=["HITL"])


class DecisionRequest(BaseModel):
    reviewer_id: str
    decision: str
    notes: str = ""
    override_ai: bool = False
    settlement_override_usd: float | None = None


def _row_to_summary(r: sqlite3.Row) -> dict:
    return {
        "ticket_id": r["ticket_id"],
        "claim_id": r["claim_id"],
        "priority": r["priority"],
        "priority_score": r["priority_score"],
        "status": r["status"],
        "created_at": r["created_at"],
        "sla_deadline": r["sla_deadline"],
        "triggers": json.loads(r["triggers"]),
    }


@router.get("/queue")
def get_queue(status: str = "pending", _: object = Depends(get_current_user)):
    """Fetches all HITL tickets with the given status (pending or resolved)."""

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM hitl_queue WHERE status = ? ORDER BY priority_score DESC, created_at ASC",
            (status,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_summary(r) for r in rows]


@router.get("/ticket/{ticket_id}")
def get_ticket(ticket_id: str, _: object = Depends(get_current_user)):
    """Fetches detailed info for a specific HITL ticket, including the state snapshot at the time of 
    pause and the reviewer's decision/notes if resolved."""
    
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return {
        **_row_to_summary(row),
        "review_brief": row["review_brief"],
        "state_snapshot": json.loads(row["state_snapshot"]) if row["state_snapshot"] else {},
        "resolved_at": row["resolved_at"],
        "reviewer_id": row["reviewer_id"],
        "human_decision": row["human_decision"],
        "human_notes": row["human_notes"],
    }


@router.post("/decide/{ticket_id}")
def decide(
    ticket_id: str,
    body: DecisionRequest,
    user=Depends(require_role("reviewer", "admin")),
):
    """
    Approver submits a decision for a paused claim. Two things happen atomically:
      1. HITL ticket is marked resolved in the review queue.
      2. The paused LangGraph pipeline is RESUMED with the approver's decision
         (via Command(resume=...)), runs to completion, and the Claim row is
         updated with the final state.
    """
    from src.models.schemas import ClaimDecision

    valid = [d.value for d in ClaimDecision]
    if body.decision not in valid:
        raise HTTPException(status_code=400, detail=f"Decision must be one of {valid}")

    # 1. Resolve the review-queue ticket + fetch claim_id for resume.
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT claim_id, status FROM hitl_queue WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if row["status"] == "resolved":
            raise HTTPException(status_code=400, detail="Ticket already resolved")
        claim_id = row["claim_id"]
        reviewer_label = body.reviewer_id or user.username
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE hitl_queue
            SET status = 'resolved', resolved_at = ?, reviewer_id = ?,
                human_decision = ?, human_notes = ?, override_ai = ?
            WHERE ticket_id = ?
            """,
            (now, reviewer_label, body.decision, body.notes,
             1 if body.override_ai else 0, ticket_id),
        )
        conn.commit()
    finally:
        conn.close()

    # 2. Resume the paused pipeline with the approver's decision.
    from api.routes_claims import resume_pipeline_for_claim

    decision_payload = {
        "decision": body.decision,
        "reviewer_id": reviewer_label,
        "notes": body.notes,
        "override_ai": body.override_ai,
        "settlement_override_usd": body.settlement_override_usd,
    }
    try:
        claim_snapshot = resume_pipeline_for_claim(claim_id, decision_payload)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Resume failed for %s", claim_id)
        raise HTTPException(status_code=500, detail=f"Pipeline resume failed: {e}")

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "decision": body.decision,
        "reviewer": reviewer_label,
        "claim_id": claim_id,
        "status": "processing",  # Pipeline is resuming in background
    }


@router.get("/stats")
def stats(_: object = Depends(get_current_user)):
    """Returns summary stats about the HITL queue, e.g. how many pending tickets total/critical/high, 
    how many resolved today, etc."""
    
    conn = _get_db()
    try:
        def one(q, *args):
            return conn.execute(q, args).fetchone()[0]
        pending = one("SELECT COUNT(*) FROM hitl_queue WHERE status='pending'")
        critical = one("SELECT COUNT(*) FROM hitl_queue WHERE status='pending' AND priority='critical'")
        high = one("SELECT COUNT(*) FROM hitl_queue WHERE status='pending' AND priority='high'")
        resolved_today = one(
            "SELECT COUNT(*) FROM hitl_queue WHERE status='resolved' AND DATE(resolved_at)=DATE('now')"
        )
        overrides = one(
            "SELECT COUNT(*) FROM hitl_queue WHERE override_ai=1 AND DATE(resolved_at)=DATE('now')"
        )
    finally:
        conn.close()
    return {
        "pending_total": pending,
        "pending_critical": critical,
        "pending_high": high,
        "resolved_today": resolved_today,
        "human_overrides_today": overrides,
    }
