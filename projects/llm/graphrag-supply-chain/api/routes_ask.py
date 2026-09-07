"""Routing for question answering.

THIN BY DESIGN. Each handler does exactly three things:

    1. receive a validated request (Pydantic enforces the shape)
    2. call ONE service method
    3. map the domain result to a response model, or a domain exception to a
       status code

There is no orchestration here, no aggregation, no guardrail logic and no
Cypher. All of it lives in `src/services/qa.py`, which is why the same pipeline
is reachable from the CLI and a notebook without a web server.

The exception mapping is the routing layer's real job: only this layer knows
what HTTP is. `GuardrailViolation` becomes 400 (or 429 for a rate limit) and
`BudgetExceeded` becomes 429 - decisions a service has no business making.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.guardrails import GuardrailViolation
from src.guardrails.limits import BudgetExceeded
from src.services import AnswerBundle, QAService

from .deps import Services, get_services
from .models import (AskRequest, AskResponse, CompareRequest, CompareResponse,
                     EvidenceOut, GraphEntityOut, RetrievalOut)
from .security import require_api_key

router = APIRouter(prefix="/api", tags=["ask"])


def _to_response(bundle: AnswerBundle) -> AskResponse:
    """Domain object -> wire format. The only mapping this module performs."""
    retrieval = bundle.retrieval
    return AskResponse(
        question=bundle.answer.question,
        strategy=retrieval.strategy,
        answer=bundle.answer.text,
        retrieval=RetrievalOut(
            strategy=retrieval.strategy,
            label=bundle.label,
            trace=retrieval.trace,
            # De-duplicated: a hybrid run legitimately executes the same query
            # more than once, and showing it three times teaches nothing.
            cypher_run=list(dict.fromkeys(retrieval.cypher_run)),
            evidence=[
                EvidenceOut(
                    kind=e.kind, text=e.text, source_id=e.source_id,
                    doc_id=e.doc_id, title=e.title, doc_type=e.doc_type,
                    score=e.score, retrieved_by=e.retrieved_by,
                )
                for e in retrieval.evidence
            ],
            entities=[
                GraphEntityOut(
                    key=e.key, name=e.name, type=e.type, hops=e.hops,
                    path_names=e.path_names, path_rels=e.path_rels,
                )
                for e in retrieval.entities
            ],
            latency_ms=round(retrieval.latency_ms, 2),
            stats=retrieval.stats,
        ),
        metrics=bundle.answer.as_dict(),
        usage=bundle.usage,
        guardrails=bundle.guardrails.as_dict() if bundle.guardrails else {},
        validation=bundle.answer.validation,
    )


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GuardrailViolation):
        code = (status.HTTP_429_TOO_MANY_REQUESTS if exc.kind == "rate_limit"
                else status.HTTP_400_BAD_REQUEST)
        return HTTPException(
            status_code=code,
            detail={"message": str(exc), "kind": exc.kind, "detail": exc.detail},
        )
    if isinstance(exc, BudgetExceeded):
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                             detail={"message": str(exc), "kind": "budget"})
    raise exc


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, caller: str = Depends(require_api_key),
        services: Services = Depends(get_services)) -> AskResponse:
    try:
        bundle = services.qa.ask(
            request.question, request.strategy, caller=caller,
            include_trace=request.include_trace,
        )
    except (GuardrailViolation, BudgetExceeded) as exc:
        raise _http_error(exc) from exc
    return _to_response(bundle)


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest, caller: str = Depends(require_api_key),
            services: Services = Depends(get_services)) -> CompareResponse:
    try:
        result = services.qa.compare(request.question, list(request.strategies),
                                     caller=caller)
    except (GuardrailViolation, BudgetExceeded) as exc:
        raise _http_error(exc) from exc
    return CompareResponse(
        question=result.question,
        results=[_to_response(b) for b in result.bundles],
        comparison=result.comparison,
        document_matrix=result.document_matrix,
    )


@router.get("/strategies")
def strategies() -> list[dict[str, str]]:
    """What each strategy does, so the client renders it without hardcoding."""
    return QAService.strategy_catalogue()
