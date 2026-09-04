"""FastAPI application.

ALL business logic lives behind this API. The Streamlit app is an HTTP client
and holds none of it.

That is an architectural decision, not a style preference, and it buys three
things:

  1. Guardrails cannot be bypassed. A control enforced in a frontend is a
     control anyone can skip with curl. Rate limiting, injection scanning,
     budget caps and output validation all run in the route handler, so every
     client gets them whether it wants them or not.
  2. The expensive singletons live in one process. A Neo4j driver holds a
     connection pool and the LLM client holds the embedding cache; Streamlit
     re-runs its whole script on every widget interaction, which is exactly the
     wrong lifetime for either.
  3. The UI is replaceable. A React frontend, a CLI, or a notebook all talk to
     the same contract, and swapping one does not put correctness at risk.

Interactive docs at http://localhost:8000/docs once running.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                            # noqa: E402
from src.graph.client import GraphUnavailable                 # noqa: E402
from src.guardrails import GuardrailViolation                 # noqa: E402

from .deps import build_services, get_services, set_services  # noqa: E402
from .routes_ask import router as ask_router                  # noqa: E402
from .routes_graph import router as graph_router              # noqa: E402
from .routes_guardrails import router as guardrails_router    # noqa: E402
from .routes_jobs import router as jobs_router                # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
)
log = logging.getLogger("graphrag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        services = build_services()
    except (ConfigError, GraphUnavailable) as exc:
        # Fail at startup with the actionable message, rather than starting
        # successfully and returning 500s that say "NoneType has no attribute".
        log.error("Startup failed.\n\n%s\n", exc)
        raise

    set_services(services)

    if not os.getenv("API_KEY"):
        log.warning(
            "API_KEY is not set, so this API is UNAUTHENTICATED. That is fine "
            "on localhost and not fine anywhere else: an open endpoint here "
            "spends money on a third-party model API for whoever finds it. Set "
            "API_KEY in .env before binding to any interface but 127.0.0.1."
        )

    counts = services.graph.counts()
    if counts:
        log.info("Graph: %d nodes, %d relationships",
                 sum(v for k, v in counts.items() if k.startswith("node:")),
                 sum(v for k, v in counts.items() if k.startswith("rel:")))
    else:
        log.warning("The graph is empty. POST /api/ingest or run "
                    "`python run.py ingest`.")

    yield

    services.graph.close()
    set_services(None)


app = FastAPI(
    title="GraphRAG Supply Chain Intelligence",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)

# Locked to localhost origins by default. A wildcard here on a service that
# spends money per request is a way to let any page on the internet spend it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in os.getenv(
            "API_CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(",") if o.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(GuardrailViolation)
async def guardrail_handler(request: Request, exc: GuardrailViolation):
    """Any guardrail that escapes a route handler still returns a structured,
    explicable refusal rather than a 500."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": {"message": str(exc), "kind": exc.kind,
                            "detail": exc.detail}},
    )


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness plus a real readiness signal.

    Reports whether the graph is populated, because "the process is up" and
    "the system can answer a question" are different states and conflating them
    is how a health check passes while every request fails.
    """
    try:
        services = get_services()
        counts = services.graph.counts()
        nodes = sum(v for k, v in counts.items() if k.startswith("node:"))
        return {
            "status": "ok",
            "graph_populated": bool(counts),
            "nodes": nodes,
            "model": services.config.llm["model"],
            "embedding_dimensions": services.config.embedding["dimensions"],
            "guardrails_enabled": services.guard.enabled,
            "authenticated": bool(os.getenv("API_KEY")),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "error": str(exc)}


app.include_router(ask_router)
app.include_router(graph_router)
app.include_router(jobs_router)
app.include_router(guardrails_router)
