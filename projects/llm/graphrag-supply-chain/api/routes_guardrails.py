"""Routing for guardrail inspection. Thin: validate, delegate, map.

The scanners and the policy live in `src/guardrails/`; assembling them into an
inspectable view lives in `src/services/security_service.py`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .deps import Services, get_services
from .models import ScanRequest, ScanResponse
from .security import require_api_key

router = APIRouter(prefix="/api/guardrails", tags=["guardrails"],
                   dependencies=[Depends(require_api_key)])


@router.get("/config")
def config(services: Services = Depends(get_services)) -> dict[str, Any]:
    return services.security.policy()


@router.post("/scan", response_model=ScanResponse)
def scan(request: ScanRequest,
         services: Services = Depends(get_services)) -> ScanResponse:
    return ScanResponse(**services.security.scan(request.text,
                                                 as_document=request.as_document))


@router.get("/audit")
def audit(limit: int = Query(100, ge=1, le=1000),
          event: str | None = Query(None),
          services: Services = Depends(get_services)) -> dict[str, Any]:
    return services.security.audit(limit=limit, event=event)


@router.get("/adversarial-sample")
def adversarial_sample(services: Services = Depends(get_services)) -> dict[str, Any]:
    sample = services.security.adversarial_sample()
    if sample is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No adversarial sample found.")
    return sample
