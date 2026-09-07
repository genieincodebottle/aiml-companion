"""Routing for graph exploration. Thin: validate, delegate, map.

All graph logic lives in `src/services/graph_service.py`. This module knows
about HTTP status codes and nothing else.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.graph import queries

from .deps import Services, get_services
from .models import (CensusResponse, CookbookEntry, CypherRequest,
                     CypherResponse, EntityDetail, EntitySummary,
                     SubgraphResponse)
from .security import require_api_key

router = APIRouter(prefix="/api/graph", tags=["graph"],
                   dependencies=[Depends(require_api_key)])


@router.get("/census", response_model=CensusResponse)
def census(services: Services = Depends(get_services)) -> CensusResponse:
    return CensusResponse(**services.graph.census())


@router.get("/schema")
def schema(services: Services = Depends(get_services)) -> dict:
    return services.graph.schema()


@router.get("/entities", response_model=list[EntitySummary])
def entities(entity_type: str | None = Query(None),
             search: str | None = Query(None, max_length=120),
             limit: int = Query(500, ge=1, le=2000),
             services: Services = Depends(get_services)) -> list[EntitySummary]:
    rows = services.graph.entities(entity_type=entity_type, search=search,
                                   limit=limit)
    return [EntitySummary(**row) for row in rows]


@router.get("/entity/{key:path}", response_model=EntityDetail)
def entity_detail(key: str,
                  services: Services = Depends(get_services)) -> EntityDetail:
    detail = services.graph.entity_detail(key)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"No entity with key '{key}'")
    return EntityDetail(**detail)


@router.get("/subgraph", response_model=SubgraphResponse)
def subgraph(key: list[str] = Query(...),
             hops: int = Query(2, ge=1, le=queries.MAX_ALLOWED_HOPS),
             limit: int = Query(60, ge=1, le=300),
             services: Services = Depends(get_services)) -> SubgraphResponse:
    return SubgraphResponse(**services.graph.subgraph(key, hops=hops, limit=limit))


@router.get("/cookbook", response_model=list[CookbookEntry])
def cookbook(services: Services = Depends(get_services)) -> list[CookbookEntry]:
    return [CookbookEntry(**entry) for entry in services.graph.cookbook()]


@router.post("/cypher", response_model=CypherResponse)
def run_cypher(request: CypherRequest,
               services: Services = Depends(get_services)) -> CypherResponse:
    try:
        return CypherResponse(**services.graph.run_cypher(request.cypher,
                                                          request.parameters))
    except ValueError as exc:
        # The service rejected a write. It raised ValueError because it does not
        # know what a status code is; deciding that means 400 is this layer's job.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface the database's own message
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
