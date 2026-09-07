"""Phase III over HTTP: generate, judge, revise - or drop."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.services import ServingService

from .deps import get_context, run, serving_service
from .models import Result, ServeRequest

router = APIRouter(prefix="/api/serving", tags=["phase III - deployment"])


@router.post("/serve", response_model=Result)
def serve(body: ServeRequest, service: ServingService = Depends(serving_service)) -> Result:
    data = run(service.serve, body.record_id, max_retries=body.max_retries)
    return Result(data=data, provenance=get_context().runtime.provenance())


@router.post("/serve-all", response_model=Result)
def serve_all(
    limit: int | None = Query(default=None, ge=1, le=500),
    service: ServingService = Depends(serving_service),
) -> Result:
    return Result(
        data=run(service.serve_all, limit=limit),
        provenance=get_context().runtime.provenance(),
    )


@router.get("/retry-curve", response_model=Result)
def retry_curve(
    max_k: int = Query(default=6, ge=0, le=12),
    service: ServingService = Depends(serving_service),
) -> Result:
    return Result(
        data=run(service.retry_curve, max_k=max_k),
        provenance=get_context().runtime.provenance(),
    )
