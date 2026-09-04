"""Routing for long-running jobs. Thin: validate, delegate, map.

Job orchestration - the worker thread, the one-at-a-time lock, the progress
state - lives in `src/services/jobs.py`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.services import JobBusy

from .deps import Services, get_services
from .models import EvalRequest, EvalStatus, IngestRequest, IngestStatus
from .security import require_api_key

router = APIRouter(prefix="/api", tags=["jobs"],
                   dependencies=[Depends(require_api_key)])


@router.post("/ingest", response_model=IngestStatus)
def start_ingest(request: IngestRequest, caller: str = Depends(require_api_key),
                 services: Services = Depends(get_services)) -> IngestStatus:
    try:
        return IngestStatus(**services.jobs.start_ingest(reset=request.reset,
                                                         caller=caller))
    except JobBusy as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/ingest/status", response_model=IngestStatus)
def ingest_status(services: Services = Depends(get_services)) -> IngestStatus:
    return IngestStatus(**services.jobs.ingest_status())


@router.post("/eval", response_model=EvalStatus)
def start_eval(request: EvalRequest, caller: str = Depends(require_api_key),
               services: Services = Depends(get_services)) -> EvalStatus:
    try:
        return EvalStatus(**services.jobs.start_eval(
            strategies=list(request.strategies) if request.strategies else None,
            question_ids=request.question_ids, judge=request.judge, caller=caller,
        ))
    except JobBusy as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/eval/status", response_model=EvalStatus)
def eval_status(services: Services = Depends(get_services)) -> EvalStatus:
    return EvalStatus(**services.jobs.eval_status())


@router.get("/eval/questions")
def golden_questions(services: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return services.jobs.golden_questions()
