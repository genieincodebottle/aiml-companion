"""FastAPI entrypoint for the Mission Control platform.

Run:  uv run uvicorn main:app --reload --port 8000

The server always starts. With a Gemini key the gateway routes real model calls
(`live`); without one it uses the deterministic offline provider (`offline`). The one
exception is MC_MODE=live with no key: starting an investigation then returns 503,
because a deployment that asked for a real model should fail loudly, not run a stand-in.

Who is acting:
  - MC_OPERATORS set: every write needs `Authorization: Bearer <token>`; the token
    names the operator and their roles (analyst opens cases, reviewer decides them).
  - MC_OPERATORS unset (local default): login is off. The console says who is acting
    in an `X-Operator` header and nothing verifies it. Maker-checker still applies.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import get_settings
from src.control_plane import get_control_plane
from src.models.events import EventLog
from src.runtime import AlreadyDecided, MakerCheckerViolation, NotAwaitingApproval
from src.runtime.engine import TERMINAL

# Read once at import, and only for what must exist before the first request (log
# level, CORS). Request-scoped settings resolve lazily, because tests change the
# environment after import; an import-time singleton once made the "offline" test
# suite bill a real API on any machine with a key exported.
_boot = get_settings()
logging.basicConfig(level=_boot.log_level)
logger = logging.getLogger("mission-control")

app = FastAPI(title="Mission Control - Agent Control Plane", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_boot.cors_origins),  # never "*": the API accepts credentials
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Operator"],
)

# Investigations run here, off the event loop, so a slow live case never blocks health
# checks or the event stream. Bounded: a burst of starts queues instead of spawning
# unbounded threads.
_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mc-case")
_NAME = re.compile(r"^[A-Za-z0-9._@-]{1,40}$")
_TERMINAL_VALUES = {s.value for s in TERMINAL}


def _plane():
    return get_control_plane()


def _settings():
    return get_settings()


# --- who is acting ------------------------------------------------------------
def operator(request: Request) -> dict:
    """The person behind this request: {"name", "roles", "verified"}."""
    s = _settings()
    if s.login_required:
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        for op in s.operators:
            if token and hmac.compare_digest(token, op.token):
                return {"name": op.name, "roles": sorted(op.roles), "verified": True}
        raise HTTPException(status_code=401, detail="a valid operator token is required (Authorization: Bearer ...)")
    name = request.headers.get("x-operator", "").strip() or "local-user"
    if not _NAME.match(name):
        raise HTTPException(status_code=400, detail="X-Operator must be 1-40 letters, digits, '.', '_', '@' or '-'")
    return {"name": name, "roles": ["analyst", "reviewer"], "verified": False}


def require_role(role: str):
    def _check(op: dict = Depends(operator)) -> dict:
        if role not in op["roles"]:
            raise HTTPException(status_code=403, detail=f"{op['name']} does not have the {role} role")
        return op

    return _check


# --- request models ---------------------------------------------------------
class TriggerRequest(BaseModel):
    alert_id: str = Field(max_length=32)
    inject_commit_fault: bool = False


class DecisionRequest(BaseModel):
    workflow_id: str = Field(max_length=80)
    decision: str  # "approve" | "reject"
    note: str | None = Field(default=None, max_length=500)


# --- routes -----------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    s = _settings()
    live = s.use_live_provider
    if not s.can_run:
        detail = "MC_MODE=live but no Gemini key is set: investigations are refused until GOOGLE_API_KEY is set."
    elif live:
        detail = "Live Gemini through the gateway."
    else:
        detail = "Offline mode: a deterministic stand-in answers every model call. Set GOOGLE_API_KEY to go live."
    return {
        "status": "ok" if s.can_run else "degraded",
        "mode": s.execution_mode,
        "requested_mode": s.mode,
        "can_run": s.can_run,
        "premium_model": s.premium_model if live else None,
        "cheap_model": s.cheap_model if live else None,
        "policy_enabled": not s.disable_policy,
        "auto_approve": s.auto_approve,
        "login_required": s.login_required,
        "tracing": _plane().tracing,
        "detail": detail,
    }


@app.get("/api/me")
async def me(op: dict = Depends(operator)) -> dict:
    return {"operator": op, "login_required": _settings().login_required}


@app.get("/api/alerts")
async def alerts() -> dict:
    return {"alerts": [a.model_dump() for a in _plane().list_alerts()]}


@app.get("/api/agents")
async def agents() -> dict:
    return {"agents": [c.model_dump() for c in _plane().list_agent_cards()]}


@app.get("/api/prompts")
async def prompts() -> dict:
    return {"prompts": _plane().prompts.summary()}


@app.post("/api/investigations")
async def start_investigation(req: TriggerRequest, op: dict = Depends(require_role("analyst"))) -> dict:
    s = _settings()
    if not s.can_run:
        raise HTTPException(
            status_code=503,
            detail="MC_MODE=live but GOOGLE_API_KEY is not set. Set it in backend/.env "
            "(https://aistudio.google.com/apikey) or use MC_MODE=auto to run offline.",
        )
    plane = _plane()
    try:
        workflow_id = plane.open_investigation(
            req.alert_id, actor=op["name"], inject_commit_fault=req.inject_commit_fault
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # The id goes back now; the steps run on a worker thread and the console follows
    # them on the event stream.
    asyncio.get_running_loop().run_in_executor(_workers, plane.run_investigation, workflow_id)
    return {"workflow_id": workflow_id, "status": "running"}


@app.post("/api/decisions")
def decide(req: DecisionRequest, op: dict = Depends(require_role("reviewer"))) -> dict:
    # Sync on purpose: approving runs the commit step inline (no model call), and
    # FastAPI runs sync handlers on its threadpool.
    plane = _plane()
    try:
        result = plane.decide(req.workflow_id, req.decision, req.note, actor=op["name"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MakerCheckerViolation as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except AlreadyDecided as exc:
        prev = exc.previous
        if prev.get("decision") == req.decision:
            # A double click or a retry: same answer, nothing new happens.
            return {"workflow_id": req.workflow_id, "duplicate": True, "view": plane.view(req.workflow_id).model_dump()}
        raise HTTPException(
            status_code=409,
            detail=(
                f"already decided: {prev.get('decision')} by {prev.get('decided_by')}; cannot change to {req.decision}"
            ),
        ) from exc
    except NotAwaitingApproval as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "workflow_id": req.workflow_id,
        "duplicate": False,
        "status": result.status,
        "view": plane.view(req.workflow_id).model_dump(),
    }


@app.get("/api/investigations")
async def list_investigations() -> dict:
    return {"investigations": [v.model_dump() for v in _plane().list_views(50)]}


@app.get("/api/investigations/{workflow_id}")
async def view(workflow_id: str) -> dict:
    try:
        return _plane().view(workflow_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _events(workflow_id: str) -> list:
    runtime = _plane().runtime
    if runtime.get_run(workflow_id) is None and runtime.load_checkpoint(workflow_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown workflow {workflow_id}")
    return EventLog(workflow_id, _settings().data_dir).replay()


@app.get("/api/investigations/{workflow_id}/events")
async def events(workflow_id: str) -> dict:
    """The full event log for a workflow."""
    log = _events(workflow_id)
    return {
        "workflow_id": workflow_id,
        "events": [{"seq": e.seq, "type": e.type.value, "payload": e.payload, "ts": e.ts} for e in log],
    }


@app.get("/api/investigations/{workflow_id}/stream")
async def stream(workflow_id: str) -> EventSourceResponse:
    """The workflow's events as Server-Sent Events, closing with a `done` frame.

    Parked at the gate counts as done for the stream: nothing more happens until a
    reviewer decides, and the console reopens the stream after a decision.
    """
    _events(workflow_id)  # 404 before opening a stream for nothing

    async def event_gen():
        log = EventLog(workflow_id, _settings().data_dir)
        seen = 0
        for _ in range(3000):  # ~5 min ceiling; a live case takes well under that
            evts = log.replay()
            while seen < len(evts):
                evt = evts[seen]
                seen += 1
                yield {"event": evt.type.value, "data": json.dumps({"seq": evt.seq, "payload": evt.payload})}
            v = _plane().view(workflow_id)
            if v.status.value in _TERMINAL_VALUES or v.status.value == "awaiting_approval":
                yield {"event": "done", "data": json.dumps(v.model_dump())}
                return
            await asyncio.sleep(0.1)

    return EventSourceResponse(event_gen())
