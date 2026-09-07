"""Phase I over HTTP: what is in the benchmark, and how a rubric scores on it."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.services import BenchmarkService

from .deps import benchmark_service, get_context, run
from .models import EvalRequest, Result

router = APIRouter(prefix="/api/benchmark", tags=["phase I - birth"])


@router.get("/report", response_model=Result)
def report(service: BenchmarkService = Depends(benchmark_service)) -> Result:
    return Result(data=run(service.report), provenance=get_context().runtime.provenance())


@router.post("/evaluate", response_model=Result)
def evaluate(
    body: EvalRequest, service: BenchmarkService = Depends(benchmark_service)
) -> Result:
    data = run(service.evaluate, body.criterion, split=body.split, staged=body.staged)
    return Result(data=data, provenance=get_context().runtime.provenance())


@router.post("/synthesise/{criterion}", response_model=Result)
def synthesise(
    criterion: str, service: BenchmarkService = Depends(benchmark_service)
) -> Result:
    data = run(service.synthesise, criterion)
    return Result(data=data, provenance=get_context().runtime.provenance())
