"""Appeals routes: submit, list, get, review."""
from __future__ import annotations # for type hints of the same class

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Appeal, Claim, User, get_session
from api.security import get_current_user, require_role

router = APIRouter(prefix="/api/appeals", tags=["Appeals"])


class SubmitAppealRequest(BaseModel):
    claim_id: str
    reason: str


class ReviewAppealRequest(BaseModel):
    decision: str  # "approved" | "denied"
    reasoning: str = ""


def _appeal_to_dict(a: Appeal) -> dict:
    return {
        "id": a.id,
        "appeal_id": a.appeal_id,
        "claim_id": a.claim_id,
        "user_id": a.user_id,
        "reason": a.reason,
        "status": a.status,
        "reviewer_id": a.reviewer_id,
        "review_decision": a.review_decision,
        "review_notes": a.review_notes,
        "created_at": a.created_at,
        "reviewed_at": a.reviewed_at,
    }


@router.post("/submit")
def submit_appeal(
    body: SubmitAppealRequest,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    claim = session.exec(select(Claim).where(Claim.claim_id == body.claim_id)).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    if user.role == "user" and claim.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your claim")
    appeal = Appeal(
        appeal_id=f"APL-{uuid.uuid4().hex[:8].upper()}",
        claim_id=body.claim_id,
        user_id=user.id,
        reason=body.reason,
        status="pending",
    )
    session.add(appeal)
    session.commit()
    session.refresh(appeal)
    return _appeal_to_dict(appeal)


@router.get("/pending")
def get_pending(
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    rows = session.exec(
        select(Appeal).where(Appeal.status == "pending").order_by(Appeal.created_at.desc()).limit(limit)
    ).all()
    return {"appeals": [_appeal_to_dict(a) for a in rows]}


@router.get("/all")
def get_all(
    status: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    stmt = select(Appeal)
    if status:
        stmt = stmt.where(Appeal.status == status)
    stmt = stmt.order_by(Appeal.created_at.desc()).limit(limit)
    rows = session.exec(stmt).all()
    return {"appeals": [_appeal_to_dict(a) for a in rows]}


@router.get("/user/{user_id}")
def get_user_appeals(
    user_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if user.role == "user" and user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other users' appeals")
    rows = session.exec(
        select(Appeal).where(Appeal.user_id == user_id).order_by(Appeal.created_at.desc())
    ).all()
    return {"appeals": [_appeal_to_dict(a) for a in rows]}


@router.get("/{appeal_id}")
def get_appeal(
    appeal_id: str,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    appeal = session.exec(select(Appeal).where(Appeal.appeal_id == appeal_id)).first()
    if not appeal:
        raise HTTPException(status_code=404, detail="Appeal not found")
    if user.role == "user" and appeal.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not your appeal")
    return _appeal_to_dict(appeal)


@router.post("/{appeal_id}/review")
def review_appeal(
    appeal_id: str,
    body: ReviewAppealRequest,
    session: Session = Depends(get_session),
    reviewer: User = Depends(require_role("reviewer", "admin")),
):
    appeal = session.exec(select(Appeal).where(Appeal.appeal_id == appeal_id)).first()
    if not appeal:
        raise HTTPException(status_code=404, detail="Appeal not found")
    if body.decision not in ("approved", "denied"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'denied'")
    appeal.status = body.decision
    appeal.review_decision = body.decision
    appeal.review_notes = body.reasoning
    appeal.reviewer_id = reviewer.id
    appeal.reviewed_at = datetime.now(timezone.utc).isoformat()
    session.add(appeal)
    session.commit()
    session.refresh(appeal)
    return _appeal_to_dict(appeal)
