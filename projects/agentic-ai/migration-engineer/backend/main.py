"""FastAPI entrypoint for the Autonomous Migration Engineer backend.

Run:  uvicorn main:app --reload --port 8000

Note the deliberate difference from a raw-API agent: this server NEVER refuses to run.
When ANTHROPIC_API_KEY + the SDK are present it drives the real Claude Agent SDK worker
(`live-sdk` mode); otherwise it drives the deterministic offline worker (`stub` mode),
so reviewers can run the whole thing - fan-out, guardrails, approval gate, evals - with
zero credentials. The health endpoint reports which mode is active.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.config import get_settings
from src.orchestrator.engine import get_manager
from src.rulebook.rules import list_jobs

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("migration-engineer")

app = FastAPI(title="Autonomous Migration Engineer", version="0.1.0")

# Never fall back to "*": with allow_credentials=True that would let any website
# drive the unauthenticated run/approve endpoints on localhost (real PRs pushed).
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = get_manager()


# --- request models ---------------------------------------------------------


class TriggerRequest(BaseModel):
    job_id: str
    auto_approve: bool | None = None


class ApprovalRequest(BaseModel):
    repo_id: str
    decision: str  # "approve" | "reject"
    note: str | None = None


# --- routes -----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "mode": settings.execution_mode,
        "model": (
            settings.model_name
            if settings.use_real_sdk
            else settings.gemini_model if settings.use_gemini else None
        ),
        "sdk_available": settings.sdk_available,
        "github_configured": bool(settings.github_token),
        "detail": (
            "Live Claude Agent SDK worker."
            if settings.use_real_sdk
            else "Live Gemini function-calling worker."
            if settings.use_gemini
            else "Running the deterministic offline worker (no API key / SDK, or ME_FORCE_STUB=1)."
        ),
    }


@app.get("/api/jobs")
async def jobs() -> dict:
    return {"jobs": list_jobs()}


@app.post("/api/runs")
async def create_run(req: TriggerRequest) -> dict:
    try:
        state = manager.trigger(req.job_id, req.auto_approve)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": state.job_id}


@app.get("/api/runs")
async def list_runs() -> dict:
    return {"runs": manager.list()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    state = manager.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="run not found")
    return state.model_dump(mode="json")


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    if manager.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")

    async def event_generator():
        async for event in manager.subscribe(run_id):
            yield {"data": json.dumps(event.model_dump(mode="json"))}

    return EventSourceResponse(event_generator())


@app.post("/api/runs/{run_id}/approve")
async def approve_repo(run_id: str, req: ApprovalRequest) -> dict:
    if manager.get(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    if req.decision not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
    ok = manager.resolve_approval(run_id, req.repo_id, req.decision, req.note or "")
    if not ok:
        raise HTTPException(status_code=409, detail="no pending approval for that repo_id")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
