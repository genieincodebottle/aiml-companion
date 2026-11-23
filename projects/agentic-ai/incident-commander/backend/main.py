"""FastAPI entrypoint for the AI Incident Commander backend.

Run:  uvicorn main:app --reload --port 8000

The API surface is small on purpose - trigger, stream, approve, cancel - because
the interesting behaviour lives in the loop. What is worth reading here is the
middleware and dependency layer, which covers the boring production concerns that
agent demos usually skip:

- **Authentication** (`IC_API_KEY`). Off by default for local dev, because a
  learner should not have to fight auth to see the loop run. But the approval
  endpoint decides whether a production rollback happens - shipping that with no
  auth is the kind of thing that reads fine in a demo and is indefensible in
  review, so the hook is here and documented.
- **Rate limiting** on mutating endpoints. Each incident spends real model quota;
  an unbounded trigger endpoint is a wallet-drain vector.
- **Idempotency** on approvals. A double-clicked Approve button must not race.
- **`Last-Event-ID` resume** on the SSE stream, so a reconnect does not replay the
  whole timeline.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agents import PROMPT_VERSION
from src.auth import (
    ROLES,
    authenticate,
    create_token,
    create_user,
    current_user,
    get_user_by_username,
    require_role,
    require_user,
    seed_admin_if_empty,
)
from src.config import get_settings
from src.db import db_ready, dispose_db, init_db
from src.db import repository as repo
from src.llm import token_tracker
from src.loop import get_manager
from src.memory.runbook import get_memory
from src.sandbox.scenarios import list_scenarios

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("incident-commander")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bring up optional persistence. Never fatal: no database is a valid setup."""
    if await init_db():
        await seed_admin_if_empty()
        get_manager().set_persistence(repo)
    yield
    await dispose_db()


app = FastAPI(
    lifespan=lifespan,
    title="AI Incident Commander",
    version="2.0.0",
    description="Autonomous SRE multi-agent system with a bounded, human-supervised agent loop.",
)

# No wildcard fallback, ever. `allow_origins=["*"]` with `allow_credentials=True`
# makes Starlette echo the request's Origin header back with credentials allowed,
# which lets any web page a logged-in operator visits call the approval endpoint
# with their cookie. An empty IC_CORS_ORIGINS now means "no cross-origin callers",
# which is the safe reading of an empty allowlist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = get_manager()


# --- cross-cutting concerns -------------------------------------------------


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Shared-secret auth. No-op when `IC_API_KEY` is unset (local dev default)."""
    api_key = get_settings().api_key
    if not api_key:
        return
    # compare_digest, not ==: a shared secret checked with an early-exit string
    # compare leaks its prefix length to a timing probe.
    if not secrets.compare_digest(x_api_key or "", api_key):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


async def rate_limit(request: Request) -> None:
    """Fixed-window-per-minute limiter on mutating endpoints, keyed by client IP.

    In-process and therefore per-replica: correct for this single-process demo,
    but a real deployment behind more than one replica needs a shared store
    (Redis) or the effective limit becomes `limit * replicas`. Said out loud here
    because "we added a rate limiter" that silently does not hold under scale-out
    is a very common and very quiet bug.
    """
    limit = get_settings().rate_limit_per_min
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    # Opportunistic pruning: without it the dict keeps one entry per client IP ever
    # seen, for the life of the process - a slow leak an attacker can drive by
    # rotating source addresses. Checked by size so the sweep is amortized-free.
    if len(_rate_buckets) > 1024:
        for key in [k for k, b in _rate_buckets.items() if not b or now - b[-1] > 60]:
            del _rate_buckets[key]
    bucket = _rate_buckets[client]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail=f"rate limit exceeded ({limit}/min)")
    bucket.append(now)


def _actor(user: dict | None) -> str:
    """The name recorded in the audit trail. "anonymous" when login is off."""
    return (user or {}).get("username") or "anonymous"


def require_llm() -> None:
    """Refuse only when Gemini was forced (IC_LLM_MODE=gemini) and there is no key."""
    if not get_settings().can_run_incidents:
        raise HTTPException(
            status_code=503,
            detail="IC_LLM_MODE=gemini but GOOGLE_API_KEY is not configured. Set GOOGLE_API_KEY in "
            "backend/.env (free at https://aistudio.google.com/apikey), or unset IC_LLM_MODE to use "
            "the offline agent.",
        )


# --- request models ---------------------------------------------------------


class TriggerRequest(BaseModel):
    scenario_id: str = Field(max_length=64)
    #: Lands in the agents' trusted context, so it is held to a service-name shape.
    service: str | None = Field(default=None, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    #: Per-run opt-in for LOW/MEDIUM risk actions only. High risk always stops for
    #: a human (GuardrailManager.requires_approval). Only operators and admins can
    #: start an incident, so only they can make this choice.
    auto_approve: bool | None = None


class ApprovalRequest(BaseModel):
    action_id: str
    decision: str = Field(description="approve | reject")
    note: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    # bcrypt only reads 72 bytes and bcrypt>=5 raises on longer input.
    password: str = Field(min_length=8, max_length=72)
    email: str | None = None
    role: str = "operator"


# --- auth routes ------------------------------------------------------------


def _require_db() -> None:
    if not db_ready():
        raise HTTPException(
            status_code=503,
            detail="Accounts need Postgres. Set IC_DATABASE_URL (docker compose brings one up) and restart.",
        )


@app.post("/api/auth/login", dependencies=[Depends(rate_limit)])
async def login(body: LoginRequest, response: Response) -> dict:
    """Exchange credentials for a session token.

    One message for every failure: distinguishing "no such user" from "wrong
    password" turns this endpoint into an account enumeration oracle.

    The token is returned in the body (for the Authorization header) AND set as an
    HttpOnly, SameSite=Strict cookie, because the browser's EventSource cannot
    send headers and the live stream must not be the one unauthenticated route.
    """
    _require_db()
    user = await authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="incorrect username or password")
    token, expires_in = create_token(user)
    response.set_cookie("ic_token", token, max_age=expires_in, httponly=True, samesite="strict", path="/api")
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in, "user": user.to_public()}


@app.post("/api/auth/register", dependencies=[Depends(rate_limit)])
async def register(body: RegisterRequest, actor: dict = Depends(require_role("admin"))) -> dict:
    """Create an account. Admin-only.

    Self-service signup is deliberately absent: this console can approve a
    production rollback, so "anyone may create themselves an operator account" is
    not a defensible default even in a demo.
    """
    _require_db()
    if not get_settings().auth_enabled:
        # With login off, `require_role` waves everyone through as an anonymous
        # admin - correct for demo endpoints, but here it would mean anyone who can
        # reach the API may plant a persistent admin account that stays valid after
        # IC_AUTH_REQUIRED is later turned on (and the seed admin will not fire,
        # because the table is no longer empty). Fail closed instead.
        raise HTTPException(
            status_code=403,
            detail="account creation requires IC_AUTH_REQUIRED=true; with login off, the seed admin "
            "(IC_SEED_ADMIN_USER) is the only account that should exist",
        )
    if body.role not in ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {ROLES}")
    if await get_user_by_username(body.username):
        raise HTTPException(status_code=409, detail="username already taken")
    user = await create_user(body.username, body.password, role=body.role, email=body.email)
    logger.info("user '%s' created by '%s'", user.username, actor.get("username"))
    return user.to_public()


@app.get("/api/auth/me")
async def whoami(user: dict | None = Depends(current_user)) -> dict:
    """Who the caller is, and whether logging in is required at all."""
    settings = get_settings()
    return {"user": user, "login_required": settings.auth_enabled, "roles": list(ROLES)}


@app.post("/api/auth/logout")
async def logout(response: Response) -> dict:
    """Client-side logout: the browser drops the token.

    Stated plainly because it matters - there is no server-side revocation list, so
    a token that has already been issued stays valid until it expires. Adding one
    is the first thing this would need before facing the internet.
    """
    response.delete_cookie("ic_token", path="/api")
    return {"ok": True, "detail": "discard the token client-side; it remains valid until it expires"}


# --- routes -----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    settings = get_settings()
    offline = settings.llm_mode == "offline"
    if offline:
        detail = "Offline mode: agents are a rule-based runbook, not Gemini. Add GOOGLE_API_KEY for real reasoning."
    elif not settings.llm_enabled:
        detail = "IC_LLM_MODE=gemini but GOOGLE_API_KEY is not configured; incidents are disabled."
    else:
        detail = None
    return {
        "status": "ok" if settings.can_run_incidents else "degraded",
        #: "gemini" or "offline". The console shows a banner whenever it is offline.
        "llm_mode": settings.llm_mode,
        "model": "offline" if offline else settings.model_name,
        "model_fast": None if offline else (settings.model_fast or None),
        "engine": settings.engine,
        "role": settings.agent_role,
        "llm_configured": settings.llm_enabled,
        "prompt_version": PROMPT_VERSION,
        # Two distinct locks, named to match: the shared-secret header, and login.
        # This field was previously called `auth_required`, which read as the login
        # setting and made a health check pass on a server where login was off.
        "api_key_required": bool(settings.api_key),
        "login_required": settings.auth_enabled,
        "db_configured": settings.db_configured,
        "db_connected": db_ready(),
        "detail": detail,
    }


@app.get("/api/config")
async def runtime_config() -> dict:
    """The loop's actual ceilings and safety settings, for the UI and for auditing.

    Exposed because "what are this agent's limits?" should be answerable without
    reading the source or trusting a README that may have drifted.
    """
    settings = get_settings()
    return {
        "max_iterations": settings.max_iterations,
        "max_cost_usd": settings.max_cost_usd,
        "max_wallclock_s": settings.max_wallclock_s,
        "max_evidence_rounds": settings.max_evidence_rounds,
        "min_action_confidence": settings.min_action_confidence,
        "approval_timeout_s": settings.approval_timeout_s,
        "auto_approve": settings.auto_approve,
        "agent_role": settings.agent_role,
        "critic_enabled": settings.critic_enabled,
        "supervisor_enabled": settings.supervisor_enabled,
        "memory_enabled": settings.memory_enabled,
        "quarantine_injections": settings.quarantine_injections,
        "redact_sensitive": settings.redact_sensitive,
        "engine": settings.engine,
    }


# Read endpoints require a login when one is configured (a no-op otherwise).
# The SSE stream included: login sets an HttpOnly cookie, which EventSource sends.
@app.get("/api/scenarios", dependencies=[Depends(require_user)])
async def scenarios() -> dict:
    return {"scenarios": list_scenarios()}


@app.post(
    "/api/incidents",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def create_incident(req: TriggerRequest, user: dict = Depends(require_role("operator", "admin"))) -> dict:
    """Start an incident. Operators and admins only: a viewer may watch, not act."""
    require_llm()
    try:
        state = manager.trigger(req.scenario_id, req.service, req.auto_approve, actor=_actor(user))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Stamp who started it, so the audit trail answers "who ran this?" and not just
    # "what happened?". Written once here rather than threaded through the loop,
    # which has no business knowing about users.
    if db_ready() and user.get("id"):
        await repo.save_incident(state.model_dump(mode="json"), created_by=user["id"])
    return {"incident_id": state.incident_id, "engine": state.engine}


@app.get("/api/incidents", dependencies=[Depends(require_user)])
async def list_incidents() -> dict:
    """Live incidents first, then anything persisted that is no longer in memory.

    In-memory wins on conflict: a running incident's live state is fresher than the
    last row written for it.
    """
    live = manager.list()
    seen = {i["incident_id"] for i in live}
    stored = [row for row in await repo.list_incidents() if row["incident_id"] not in seen]
    return {"incidents": live + stored, "persisted": db_ready()}


@app.get("/api/incidents/stats", dependencies=[Depends(require_user)])
async def incident_stats() -> dict:
    """Aggregates for the dashboard, computed in the database when there is one."""
    if db_ready():
        return await repo.stats()
    runs = manager.list()
    done = [r for r in runs if r.get("finalized")]
    return {
        "total": len(runs),
        "resolved": sum(1 for r in done if r["status"] == "resolved"),
        "escalated": sum(1 for r in done if r["status"] == "escalated"),
        "total_cost_usd": round(sum(r.get("total_cost_usd", 0.0) for r in runs), 6),
        "mean_iterations": round(sum(r.get("iterations", 0) for r in done) / len(done), 2) if done else 0.0,
    }


@app.get("/api/incidents/{incident_id}", dependencies=[Depends(require_user)])
async def get_incident(incident_id: str) -> dict:
    state = manager.get(incident_id)
    if state is not None:
        return state.model_dump(mode="json")
    stored = await repo.load_incident(incident_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return stored


@app.get("/api/incidents/{incident_id}/stream", dependencies=[Depends(require_user)])
async def stream_incident(
    incident_id: str,
    last_event_id: str | None = Header(default=None),
    after_seq: int = 0,
):
    """Live SSE stream of loop events.

    Honours `Last-Event-ID`: a reconnecting client resumes after the last event it
    saw rather than replaying the full history, which is what keeps a dropped
    connection invisible in the UI. The same value is also accepted as an
    `after_seq` query parameter, because the browser only sends the header on
    EventSource's own automatic reconnects - a client that deliberately
    re-subscribes (navigating back to the console mid-run) cannot set headers.
    """
    if manager.get(incident_id) is None:
        raise HTTPException(status_code=404, detail="incident not found")

    try:
        after_seq = int(last_event_id) if last_event_id else after_seq
    except ValueError:
        pass

    async def event_generator():
        async for event in manager.subscribe(incident_id, after_seq=after_seq):
            # `id` drives Last-Event-ID resume. The event NAME is deliberately left as
            # the default "message": named SSE events require the client to
            # addEventListener per type, so every new event type would be a silent
            # frontend break. The type travels inside the payload instead.
            yield {"id": str(event.seq), "data": json.dumps(event.model_dump(mode="json"))}

    return EventSourceResponse(event_generator())


@app.post(
    "/api/incidents/{incident_id}/approve",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def approve_action(
    incident_id: str,
    req: ApprovalRequest,
    # Viewers may watch an incident but may not decide one. Fail-closed, like the
    # agent's own RBAC: an unknown role matches nothing and is denied.
    user: dict = Depends(require_role("operator", "admin")),
) -> dict:
    """Resolve the human approval gate.

    Idempotent: re-submitting the same decision for the same action returns 200
    with `duplicate: true` rather than 409, so a double-clicked button or a client
    retry is harmless. A *different* decision for an already-decided action is
    still a 409, because that is a genuine conflict and silently ignoring it would
    hide a real bug.
    """
    if manager.get(incident_id) is None:
        raise HTTPException(status_code=404, detail="incident not found")
    if req.decision not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")

    # Who decided is recorded on the action, in the event stream and in the
    # postmortem: an action that can roll back production must name its approver.
    ok = manager.resolve_approval(incident_id, req.action_id, req.decision, req.note or "", actor=_actor(user))
    if ok:
        return {"ok": True, "duplicate": False}

    previous = manager.approval_already_recorded(req.action_id)
    if previous is not None:
        if previous["decision"] == req.decision:
            return {"ok": True, "duplicate": True, "decision": previous["decision"]}
        raise HTTPException(
            status_code=409,
            detail=f"action already resolved as '{previous['decision']}'; cannot change to '{req.decision}'",
        )
    raise HTTPException(status_code=409, detail="no pending approval for that action_id")


@app.post(
    "/api/incidents/{incident_id}/cancel",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def cancel_incident(incident_id: str, user: dict = Depends(require_role("operator", "admin"))) -> dict:
    """Ask a running incident to stop at its next checkpoint.

    Cooperative rather than a hard kill: an incident mid-action finishes that
    action and still writes a postmortem. An operator needs a stop button, and a
    stop button that loses the audit trail is not one.
    """
    state = manager.get(incident_id)
    if state is None:
        raise HTTPException(status_code=404, detail="incident not found")
    cancelled = manager.request_cancel(incident_id, actor=_actor(user))
    if not cancelled:
        return {"ok": False, "detail": f"incident already terminal (status={state.status.value})"}
    return {"ok": True, "status": "cancelling"}


@app.get("/api/incidents/{incident_id}/usage", dependencies=[Depends(require_user)])
async def incident_usage(incident_id: str) -> dict:
    """Per-agent token, cost and latency breakdown for one incident.

    "The incident cost $0.008" is not actionable. "The Diagnostician is 71% of that
    because it re-reads the whole telemetry bundle every round" is.
    """
    state = manager.get(incident_id)
    if state is None:
        raise HTTPException(status_code=404, detail="incident not found")
    tokens, cost = token_tracker.incident_cost(incident_id)
    return {
        "incident_id": incident_id,
        "total_tokens": tokens,
        "total_cost_usd": round(cost, 6),
        "budget_usd": state.max_cost_usd,
        "budget_used_pct": round(100 * cost / state.max_cost_usd, 2) if state.max_cost_usd else 0.0,
        "iterations": state.iterations,
        "evidence_rounds": state.evidence_rounds,
        "by_agent": token_tracker.incident_breakdown(incident_id),
    }


@app.get("/api/memory", dependencies=[Depends(require_user)])
async def memory_stats(limit: int = 20) -> dict:
    """What the agent has learned across incidents.

    Inspectable on purpose: an agent whose memory you cannot read is an agent whose
    behaviour you cannot explain.
    """
    memory = get_memory()
    limit = max(1, min(limit, 200))  # `[-0:]` would return everything
    episodes = memory.all()[-limit:]
    return {
        "stats": memory.stats(),
        "episodes": [
            {
                "episode_id": e.episode_id,
                "scenario_id": e.scenario_id,
                "service": e.service,
                "root_cause": e.root_cause,
                "effective_actions": e.effective_actions,
                "outcome": e.outcome,
                "lesson": e.lesson,
                "recorded_at": e.recorded_at,
            }
            for e in reversed(episodes)
        ],
    }


@app.get("/api/memory/search", dependencies=[Depends(require_user)])
async def memory_search(q: str, top_k: int = 3) -> dict:
    """Run the retrieval the Diagnostician runs, and show why each hit matched."""
    hits = get_memory().recall(q, top_k=top_k)
    return {"query": q, "hits": [h.as_dict() for h in hits]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
