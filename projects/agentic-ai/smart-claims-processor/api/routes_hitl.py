"""HITL routes under /api/hitl - wraps the existing queue in src/hitl/queue.py."""
from __future__ import annotations # for type hinting within the same file

import json
import logging
import sqlite3
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from api.security import require_role
from src.hitl.queue import _get_db, escalate_overdue_tickets
from src.security.audit_log import log_hitl_event

# The review queue holds claim briefs, fraud scores and reviewer notes for
# every claimant: reviewers and admins only, for reads as well as decisions.
router = APIRouter(
    prefix="/api/hitl",
    tags=["HITL"],
    dependencies=[Depends(require_role("reviewer", "admin"))],
)


class DecisionRequest(BaseModel):
    # Ignored for attribution: the audit trail records the AUTHENTICATED
    # user, not a name the browser sends. Kept optional for old clients.
    reviewer_id: str | None = None
    decision: str
    notes: str = Field("", max_length=1000)
    override_ai: bool = False
    settlement_override_usd: float | None = Field(None, ge=0)


def _row_to_summary(r: sqlite3.Row) -> dict:
    return {
        "ticket_id": r["ticket_id"],
        "claim_id": r["claim_id"],
        "priority": r["priority"],
        "priority_score": r["priority_score"],
        "status": r["status"],
        "created_at": r["created_at"],
        "sla_deadline": r["sla_deadline"],
        "sla_breached": bool(r["sla_breached"]) if "sla_breached" in r.keys() else False,
        "triggers": json.loads(r["triggers"]),
    }


@router.get("/queue")
def get_queue(status: str = "pending"):
    """Fetches all HITL tickets with the given status (pending or resolved).

    Overdue pending tickets are escalated to critical first, so they sort to
    the top of the queue."""

    conn = _get_db()
    try:
        escalate_overdue_tickets(conn)
        rows = conn.execute(
            "SELECT * FROM hitl_queue WHERE status = ? "
            "ORDER BY COALESCE(sla_breached, 0) DESC, priority_score DESC, created_at ASC",
            (status,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_summary(r) for r in rows]


@router.get("/ticket/{ticket_id}")
def get_ticket(ticket_id: str):
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
    notes = (body.notes or "").strip()
    if not notes:
        # The UI marks notes as required for the audit trail; enforce it here
        # too, so a direct API call cannot record an unexplained decision.
        raise HTTPException(status_code=400, detail="Review notes are required for the audit trail")
    if body.settlement_override_usd is not None and body.decision not in (
        ClaimDecision.APPROVED.value, ClaimDecision.APPROVED_PARTIAL.value
    ):
        raise HTTPException(status_code=400, detail="A settlement override only applies to an approval")

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
        reviewer_label = user.username
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE hitl_queue
            SET status = 'resolved', resolved_at = ?, reviewer_id = ?,
                human_decision = ?, human_notes = ?, override_ai = ?
            WHERE ticket_id = ?
            """,
            (now, reviewer_label, body.decision, notes,
             1 if body.override_ai else 0, ticket_id),
        )
        conn.commit()
    finally:
        conn.close()

    log_hitl_event(
        claim_id=claim_id,
        event="RESOLVED",
        priority="",
        triggers=[],
        reviewer_id=reviewer_label,
        human_decision=body.decision,
        human_notes=notes,
        override_ai=body.override_ai,
    )

    # 2. Resume the paused pipeline with the approver's decision.
    from api.routes_claims import resume_pipeline_for_claim

    decision_payload = {
        "decision": body.decision,
        "reviewer_id": reviewer_label,
        "notes": notes,
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
def stats():
    """Returns summary stats about the HITL queue, e.g. how many pending tickets total/critical/high,
    how many are past their SLA, how many resolved today, etc."""

    conn = _get_db()
    try:
        escalate_overdue_tickets(conn)

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
        overdue = one(
            "SELECT COUNT(*) FROM hitl_queue WHERE status='pending' AND COALESCE(sla_breached, 0)=1"
        )
    finally:
        conn.close()
    return {
        "pending_total": pending,
        "pending_sla_breached": overdue,
        "pending_critical": critical,
        "pending_high": high,
        "resolved_today": resolved_today,
        "human_overrides_today": overrides,
    }
