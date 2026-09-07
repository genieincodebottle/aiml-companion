"""FastAPI application. Four routers, one per phase.

    uvicorn api.main:app --reload --port 8000
    OFFLINE=1 uvicorn api.main:app --port 8000     # rule engine, no key, no cost

Interactive docs at /docs. The route grouping mirrors the lifecycle on purpose:
opening /docs should tell you what the four phases are before you read any of
this project's prose.
"""

from __future__ import annotations

from fastapi import FastAPI

from . import routes_benchmark, routes_monitoring, routes_serving, routes_tuning
from .deps import get_context

app = FastAPI(
    title="LLM Judge Lifecycle",
    version="1.0.0",
    description=(
        "An LLM judge treated as a maintained system rather than a static "
        "score. Birth (a labelled benchmark), Training (rubric text as the "
        "parameter), Deployment (gate plus critic, with a retry budget), and "
        "Monitoring (a drift band pegged to human disagreement)."
    ),
)

app.include_router(routes_benchmark.router)
app.include_router(routes_tuning.router)
app.include_router(routes_serving.router)
app.include_router(routes_monitoring.router)


@app.get("/api/health", tags=["meta"])
def health() -> dict[str, object]:
    """Health, plus the two caveats that have to travel with every number.

    ``offline_stub_run`` and ``single_model_config`` are surfaced here rather
    than buried in a log because they change what the results MEAN, and the
    person reading a dashboard is not the person who chose the config.
    """
    context = get_context()
    return {
        "ok": True,
        "domain": context.domain.name,
        "criteria": [
            {"id": c.id, "must_have": c.must_have} for c in context.domain.criteria
        ],
        "provenance": context.runtime.provenance(),
    }
