"""Phase II over HTTP. Tuning STAGES a rubric; promotion is a separate call."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.services import TuningService

from .deps import get_context, run, tuning_service
from .models import Result, TuneRequest

router = APIRouter(prefix="/api/tuning", tags=["phase II - training"])


@router.post("/tune", response_model=Result)
def tune(body: TuneRequest, service: TuningService = Depends(tuning_service)) -> Result:
    data = run(
        service.tune, body.criterion, reasoning_alignment=body.reasoning_alignment
    )
    return Result(data=data, provenance=get_context().runtime.provenance())


@router.post("/ablation/{criterion}", response_model=Result)
def ablation(criterion: str, service: TuningService = Depends(tuning_service)) -> Result:
    return Result(
        data=run(service.ablation, criterion),
        provenance=get_context().runtime.provenance(),
    )


@router.post("/promote/{criterion}", response_model=Result)
def promote(criterion: str, service: TuningService = Depends(tuning_service)) -> Result:
    # Separate endpoint, separate deliberate call. Promotion changes what the
    # gate rejects, which changes what reaches users, so it is never a side
    # effect of tuning.
    return Result(
        data=run(service.promote, criterion),
        provenance=get_context().runtime.provenance(),
    )
