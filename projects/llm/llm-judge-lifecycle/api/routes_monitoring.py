"""Phase IV over HTTP: the drift band, and what an alert triggers."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.services import MonitoringService

from .deps import get_context, monitoring_service, run
from .models import Result

router = APIRouter(prefix="/api/monitoring", tags=["phase IV - monitoring"])


@router.get("/weeks", response_model=Result)
def weeks(service: MonitoringService = Depends(monitoring_service)) -> Result:
    return Result(
        data={"weeks": run(service.available_weeks)},
        provenance=get_context().runtime.provenance(),
    )


@router.get("/check/{week}", response_model=Result)
def check(week: int, service: MonitoringService = Depends(monitoring_service)) -> Result:
    return Result(
        data=run(service.check, week), provenance=get_context().runtime.provenance()
    )


@router.post("/augment/{week}", response_model=Result)
def augment(week: int, service: MonitoringService = Depends(monitoring_service)) -> Result:
    # Writes to artifacts/ for review, never straight into the benchmark. A
    # pipeline that can rewrite its own ground truth unreviewed can move the
    # goalposts and pass.
    return Result(
        data=run(service.augment, week), provenance=get_context().runtime.provenance()
    )
