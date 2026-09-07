"""Claims routes: submit, list, get, process (background), status."""
from __future__ import annotations # for type hints that refer to classes defined later in the file

import json
import logging
import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Claim, User, engine, get_session
from api.security import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/claims", tags=["Claims"])


class SubmitClaimRequest(BaseModel):
    policy_number: str
    incident_type: str
    incident_date: str
    incident_description: str
    incident_location: str = ""
    police_report_number: Optional[str] = None
    estimated_amount: float
    vehicle_year: Optional[int] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    documents: list[str] = []


def _claim_to_dict(c: Claim) -> dict:
    return {
        "id": c.id,
        "claim_id": c.claim_id,
        "user_id": c.user_id,
        "policy_number": c.policy_number,
        "incident_type": c.incident_type,
        "incident_date": c.incident_date,
        "incident_description": c.incident_description,
        "incident_location": c.incident_location,
        "estimated_amount": c.estimated_amount,
        "settlement_amount": c.settlement_amount,
        "status": c.status,
        "final_decision": c.final_decision,
        "fraud_score": c.fraud_score,
        "fraud_risk_level": c.fraud_risk_level,
        "evaluation_score": c.evaluation_score,
        "decided_by": c.decided_by,
        "decided_at": c.decided_at,
        "hitl_required": c.hitl_required,
        "hitl_ticket_id": c.hitl_ticket_id,
        "agent_call_count": c.agent_call_count,
        "total_cost_usd": c.total_cost_usd,
        "processing_time_sec": c.processing_time_sec,
        "denial_reasons": json.loads(c.denial_reasons or "[]"),
        "communication_message": c.communication_message,
        "appeal_instructions": c.appeal_instructions,
        "agent_outputs": json.loads(c.agent_outputs or "{}"),
        "pipeline_path": json.loads(c.pipeline_path or "[]"),
        "error_log": json.loads(c.error_log or "[]"),
        "created_at": c.created_at,
        "completed_at": c.completed_at,
    }


def _persist_pipeline_result(claim: Claim, state: dict, elapsed: float) -> None:
    """Copy results from a terminal (non-paused) ClaimsState onto the Claim row."""

    from datetime import datetime, timezone

    decision = state.get("final_decision")
    claim.final_decision = decision.value if hasattr(decision, "value") else (str(decision) if decision else None)
    claim.settlement_amount = state.get("final_amount_usd")

    # Track who made the final decision and when
    claim.decided_at = datetime.now(timezone.utc).isoformat()
    if state.get("hitl_required") and state.get("human_reviewer_id"):
        claim.decided_by = state["human_reviewer_id"]
    else:
        claim.decided_by = "AI Agent"

    fraud = state.get("fraud_output")
    if fraud:
        claim.fraud_score = float(getattr(fraud, "fraud_score", 0.0))
        level = getattr(fraud, "fraud_risk_level", None)
        claim.fraud_risk_level = level.value if hasattr(level, "value") else (str(level) if level else None)

    evaluation = state.get("evaluation_output")
    if evaluation:
        claim.evaluation_score = float(getattr(evaluation, "overall_score", 0.0))

    claim.hitl_required = bool(state.get("hitl_required"))
    claim.hitl_ticket_id = state.get("hitl_ticket_id") or claim.hitl_ticket_id
    claim.agent_call_count = int(state.get("agent_call_count") or 0)
    claim.total_tokens_used = int(state.get("total_tokens_used") or 0)
    claim.total_cost_usd = float(state.get("total_cost_usd") or 0.0)
    claim.processing_time_sec = (claim.processing_time_sec or 0.0) + elapsed

    trace = state.get("pipeline_trace") or []
    path = [entry.get("agent") for entry in trace if isinstance(entry, dict) and entry.get("agent")]
    claim.pipeline_path = json.dumps(path)
    claim.error_log = json.dumps(state.get("error_log") or [])

    # Serialize enriched per-agent outputs for transparency
    agent_outputs = {}
    for key in ("intake_output", "fraud_output", "damage_output", "policy_output",
                "settlement_output", "evaluation_output", "communication_output"):
        obj = state.get(key)
        if obj and hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump(mode="json")
                agent_outputs[key] = dumped
            except Exception:
                agent_outputs[key] = str(obj)
    agent_outputs["_trace"] = trace  # enriched pipeline_trace with reasoning/flags
    claim.agent_outputs = json.dumps(agent_outputs, default=str)

    # Map final_decision to a user-facing status instead of always "completed"
    decision_str = claim.final_decision or ""
    _DECISION_TO_STATUS = {
        "approved": "approved",
        "approved_partial": "approved_partial",
        "denied": "denied",
        "auto_rejected": "auto_rejected",
        "fraud_investigation": "fraud_investigation",
        "pending_documents": "pending_documents",
    }
    claim.status = _DECISION_TO_STATUS.get(decision_str, "completed")

    # Persist denial reasons and communication output
    settlement = state.get("settlement_output")
    if settlement:
        claim.denial_reasons = json.dumps(getattr(settlement, "denial_reasons", None) or [])
    elif claim.final_decision == "denied":
        # No settlement output (e.g. intake rejection) - pull reasons from intake
        intake = state.get("intake_output")
        if intake:
            reasons = list(getattr(intake, "validation_flags", None) or [])
            if not reasons and getattr(intake, "intake_notes", ""):
                reasons = [intake.intake_notes]
            claim.denial_reasons = json.dumps(reasons)
    comm = state.get("communication_output")
    if comm:
        claim.communication_message = getattr(comm, "message", None)
        claim.appeal_instructions = getattr(comm, "appeal_instructions", None)

    claim.completed_at = datetime.now(timezone.utc).isoformat()


def _persist_pause(claim: Claim, interrupt_payload: dict) -> None:
    """Pipeline paused at HITL - mark the claim so approvers can act on it."""
    claim.status = "pending_human_review"
    claim.hitl_required = True
    claim.hitl_ticket_id = interrupt_payload.get("ticket_id") or claim.hitl_ticket_id


def _spawn_pipeline(claim_pk: int) -> None:
    """Run _run_pipeline in a real daemon thread.

    FastAPI's BackgroundTasks delegates sync tasks to anyio's threadpool, which
    has been unreliable here for long-running LLM work (the task silently
    vanishes and the claim sits at 'processing' forever). A plain threading
    Thread with daemon=True bypasses that entirely - the OS just runs it.
    """
    
    t = threading.Thread(target=_run_pipeline, args=(claim_pk,), daemon=True,
                         name=f"pipeline-{claim_pk}")
    t.start()
    logger.info("[claim_pk=%s] pipeline thread started (tid=%s)", claim_pk, t.ident)


def _mark_failed(claim_pk: int, error: str) -> None:
    """Last-resort status flip used when _run_pipeline crashes outside its normal try/except."""
    
    from datetime import datetime, timezone
    try:
        with Session(engine) as session:
            claim = session.get(Claim, claim_pk)
            if not claim:
                return
            claim.status = "failed"
            claim.error_log = json.dumps([error[:500]])
            claim.completed_at = datetime.now(timezone.utc).isoformat()
            session.add(claim)
            session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("[claim_pk=%s] could not even mark as failed", claim_pk)


def _run_pipeline(claim_pk: int) -> None:
    """Background task: run the LangGraph pipeline until completion OR HITL pause.

    Wrapped in a bare try/except so that ANY failure (import error, LLM key
    missing, SQLite lock, etc.) is surfaced to the backend log AND persisted
    on the claim row - otherwise the row hangs at 'processing' forever and the
    reviewer has no way to know why.
    """
    
    import traceback
    logger.info("[claim_pk=%s] _run_pipeline entered", claim_pk)

    try:
        from datetime import datetime, timezone
        from src.agents.graph import process_claim
        from src.models.state import ClaimInput
    except Exception as e:  # noqa: BLE001
        logger.exception("[claim_pk=%s] pipeline imports failed", claim_pk)
        _mark_failed(claim_pk, f"import_failed: {e}\n{traceback.format_exc()}")
        return

    try:
        with Session(engine) as session:
            claim = session.get(Claim, claim_pk)
            if not claim:
                logger.error("[claim_pk=%s] claim row vanished", claim_pk)
                return

            logger.info("[%s] pipeline starting (provider-driven agents)", claim.claim_id)
            claim.status = "processing"
            session.add(claim)
            session.commit()
            session.refresh(claim)

            pipeline_input = ClaimInput(
                claim_id=claim.claim_id,
                policy_number=claim.policy_number,
                claimant_name=claim.claimant_name or "Unknown",
                claimant_email=claim.claimant_email or "unknown@example.com",
                claimant_phone=claim.claimant_phone or "000-000-0000",
                claimant_dob=claim.claimant_dob or "1990-01-01",
                incident_date=claim.incident_date,
                incident_type=claim.incident_type,
                incident_description=claim.incident_description,
                incident_location=claim.incident_location,
                police_report_number=claim.police_report_number,
                estimated_amount=claim.estimated_amount,
                vehicle_year=claim.vehicle_year,
                vehicle_make=claim.vehicle_make,
                vehicle_model=claim.vehicle_model,
                documents=json.loads(claim.documents or "[]"),
                is_appeal=False,
                original_claim_id=None,
            )

            t0 = time.time()
            try:
                result = process_claim(pipeline_input)
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] process_claim raised", claim.claim_id)
                claim.status = "failed"
                claim.error_log = json.dumps([f"{type(e).__name__}: {e}"])
                claim.completed_at = datetime.now(timezone.utc).isoformat()
                session.add(claim)
                session.commit()
                return

            elapsed = time.time() - t0
            state = result["state"]

            if result.get("paused"):
                _persist_pause(claim, result.get("interrupt") or {})
                claim.agent_call_count = int(state.get("agent_call_count") or 0)
                claim.total_tokens_used = int(state.get("total_tokens_used") or 0)
                claim.total_cost_usd = float(state.get("total_cost_usd") or 0.0)
                claim.processing_time_sec = (claim.processing_time_sec or 0.0) + elapsed
                fraud = state.get("fraud_output")
                if fraud:
                    claim.fraud_score = float(getattr(fraud, "fraud_score", 0.0))
                    level = getattr(fraud, "fraud_risk_level", None)
                    claim.fraud_risk_level = level.value if hasattr(level, "value") else None
                # Persist agent outputs so HITL reviewers can see agent traces
                agent_outputs = {}
                for key in ("intake_output", "fraud_output", "damage_output", "policy_output",
                            "settlement_output", "evaluation_output"):
                    obj = state.get(key)
                    if obj and hasattr(obj, "model_dump"):
                        try:
                            agent_outputs[key] = obj.model_dump(mode="json")
                        except Exception:
                            agent_outputs[key] = str(obj)
                trace = state.get("pipeline_trace") or []
                agent_outputs["_trace"] = trace
                claim.agent_outputs = json.dumps(agent_outputs, default=str)
                path = [entry.get("agent") for entry in trace if isinstance(entry, dict) and entry.get("agent")]
                claim.pipeline_path = json.dumps(path)
            else:
                _persist_pipeline_result(claim, state, elapsed)

            session.add(claim)
            session.commit()
            logger.info(
                "[%s] pipeline %s in %.2fs",
                claim.claim_id,
                "paused" if result.get("paused") else "completed",
                elapsed,
            )
    except Exception as e:  # noqa: BLE001
        # Absolutely last resort - never let this task die silently.
        logger.exception("[claim_pk=%s] unhandled error in _run_pipeline", claim_pk)
        _mark_failed(claim_pk, f"unhandled: {type(e).__name__}: {e}")


def resume_pipeline_for_claim(claim_id: str, decision: dict) -> dict:
    """
    Called from the HITL decide endpoint after an approver submits a decision.
    Marks claim as 'processing' immediately, then resumes the pipeline in a
    background thread (same pattern as initial submission).
    """
    
    with Session(engine) as session:
        claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
        if not claim:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        if claim.status != "pending_human_review":
            raise HTTPException(
                status_code=400,
                detail=f"Claim {claim_id} is not awaiting review (status={claim.status})",
            )
        claim.status = "processing"
        session.add(claim)
        session.commit()

        # Run the remaining pipeline in a background thread
        t = threading.Thread(
            target=_run_resume, args=(claim.id, claim_id, decision),
            daemon=True, name=f"resume-{claim_id}",
        )
        t.start()
        logger.info("[%s] resume thread started (tid=%s)", claim_id, t.ident)

        return _claim_to_dict(claim)


def _run_resume(claim_pk: int, claim_id: str, decision: dict) -> None:
    """Background thread: resume the paused LangGraph pipeline after HITL approval."""
    
    import traceback
    logger.info("[%s] _run_resume entered", claim_id)

    try:
        from src.agents.graph import resume_claim
    except Exception as e:
        logger.exception("[%s] resume imports failed", claim_id)
        _mark_failed(claim_pk, f"import_failed: {e}\n{traceback.format_exc()}")
        return

    try:
        with Session(engine) as session:
            claim = session.get(Claim, claim_pk)
            if not claim:
                logger.error("[%s] claim row vanished during resume", claim_id)
                return

            t0 = time.time()
            try:
                result = resume_claim(claim_id, decision)
            except Exception as e:
                logger.exception("[%s] resume_claim raised", claim_id)
                claim.status = "failed"
                claim.error_log = json.dumps([f"resume_failed: {e}"])
                from datetime import datetime, timezone
                claim.completed_at = datetime.now(timezone.utc).isoformat()
                session.add(claim)
                session.commit()
                return

            elapsed = time.time() - t0
            state = result["state"]

            if result.get("paused"):
                # Shouldn't happen but handle gracefully
                _persist_pause(claim, result.get("interrupt") or {})
            else:
                _persist_pipeline_result(claim, state, elapsed)

            session.add(claim)
            session.commit()
            logger.info("[%s] resume completed in %.2fs", claim_id, elapsed)
    except Exception as e:
        logger.exception("[%s] unhandled error in _run_resume", claim_id)
        _mark_failed(claim_pk, f"unhandled_resume: {type(e).__name__}: {e}")


@router.post("/submit")
def submit_claim(
    body: SubmitClaimRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    """Submit a new claim - creates the Claim row, then launches the pipeline in a background thread."""
    
    claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    claim = Claim(
        claim_id=claim_id,
        user_id=user.id,
        policy_number=body.policy_number,
        claimant_name=user.username,
        claimant_email=user.email,
        incident_date=body.incident_date,
        incident_type=body.incident_type,
        incident_description=body.incident_description,
        incident_location=body.incident_location,
        police_report_number=body.police_report_number,
        estimated_amount=body.estimated_amount,
        vehicle_year=body.vehicle_year,
        vehicle_make=body.vehicle_make,
        vehicle_model=body.vehicle_model,
        documents=json.dumps(body.documents),
        status="submitted",
    )
    session.add(claim)
    session.commit()
    session.refresh(claim)
    logger.info("[%s] submitted by user_id=%s, launching pipeline thread", claim.claim_id, user.id)
    _spawn_pipeline(claim.id)
    return _claim_to_dict(claim)


@router.get("/all")
def get_all_claims(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    stmt = select(Claim)
    if status:
        stmt = stmt.where(Claim.status == status)
    stmt = stmt.order_by(Claim.created_at.desc()).limit(limit)
    rows = session.exec(stmt).all()
    return {"claims": [_claim_to_dict(c) for c in rows]}


@router.get("/user/{user_id}")
def get_user_claims(
    user_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if user.role == "user" and user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other users' claims")
    rows = session.exec(
        select(Claim).where(Claim.user_id == user_id).order_by(Claim.created_at.desc())
    ).all()
    return {"claims": [_claim_to_dict(c) for c in rows]}


@router.get("/{claim_id}")
def get_claim(
    claim_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    if user.role == "user" and claim.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your claim")
    return _claim_to_dict(claim)


@router.post("/{claim_id}/process")
def reprocess_claim(
    claim_id: str,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    claim.status = "processing"
    session.add(claim)
    session.commit()
    logger.info("[%s] reprocess requested, launching pipeline thread (claim_pk=%s)", claim.claim_id, claim.id)
    _spawn_pipeline(claim.id)
    return {"ok": True, "claim_id": claim_id, "status": "processing"}


@router.get("/{claim_id}/status")
def get_claim_status(
    claim_id: str,
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {
        "claim_id": claim.claim_id,
        "status": claim.status,
        "final_decision": claim.final_decision,
        "hitl_required": claim.hitl_required,
        "hitl_ticket_id": claim.hitl_ticket_id,
    }
