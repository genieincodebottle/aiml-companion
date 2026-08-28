"""FastAPI surface.

Small on purpose. The interesting code is the pipeline and the failure toggles;
this file only exposes them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import (
    BUDGETS,
    CLASSIFIER_MODEL,
    ORCHESTRATOR_MODEL,
    PRICES,
    PRICES_LAST_CHECKED,
    PROMPT_VERSIONS,
    WORKER_MODEL,
    live_mode_available,
    mode_reason,
)
from .failures import CATALOG, FailureSwitches
from .pipeline import cost_breakdown, run_request
from .tools import db

app = FastAPI(title="Multi-Agent Anatomy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLAY_DIR = Path(__file__).resolve().parents[1] / "replay"


class AskRequest(BaseModel):
    question: str = "Where is my order ORD-4412 and can I still return it?"
    tenant_id: str = "tenant-northwind"
    customer_id: str = "cust-1001"
    failures: dict[str, Any] | None = None
    reset_cache: bool = False


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "live" if live_mode_available() else "replay",
        "live_available": live_mode_available(),
        "stages": 8,
        "agents": 5,
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "models": {
            "orchestrator": ORCHESTRATOR_MODEL,
            "worker": WORKER_MODEL,
            "classifier": CLASSIFIER_MODEL,
        },
        "prices_usd_per_1m": {k: v.__dict__ for k, v in PRICES.items()},
        "prices_last_checked": PRICES_LAST_CHECKED,
        "budgets": BUDGETS.__dict__,
        "prompt_versions": PROMPT_VERSIONS,
        "failure_catalog": CATALOG,
        "mode": "live" if live_mode_available() else "replay",
        "mode_reason": mode_reason(),
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict[str, Any]:
    result = await run_request(
        req.question,
        tenant_id=req.tenant_id,
        customer_id=req.customer_id,
        switches=FailureSwitches.from_dict(req.failures),
        reset_cache=req.reset_cache,
    )
    result["cost_by_model"] = cost_breakdown(result["trace"])
    return result


@app.get("/api/replay")
def list_replays() -> list[dict[str, Any]]:
    """Recorded traces. These run with no API key at all, which is the whole
    point of shipping them."""
    out = []
    for f in sorted(REPLAY_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        out.append(
            {
                "id": f.stem,
                "title": data.get("title", f.stem),
                "teaches": data.get("teaches", ""),
                "failures": data.get("failures", {}),
                "question": data.get("question", ""),
            }
        )
    return out


@app.get("/api/replay/{replay_id}")
def get_replay(replay_id: str) -> dict[str, Any]:
    path = REPLAY_DIR / f"{replay_id}.json"
    if not path.exists():
        return {"error": "not_found", "id": replay_id}
    return json.loads(path.read_text(encoding="utf-8"))
