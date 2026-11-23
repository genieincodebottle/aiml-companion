# Incident Commander - Backend

FastAPI + the multi-agent loop + Google Gemini (or the offline agent when no key is
set). Start with the project [README](../README.md); the design is in
[docs/architecture.md](../docs/architecture.md).

## Run

Uses [uv](https://docs.astral.sh/uv/) for dependencies.

```bash
uv sync --extra dev                              # create .venv, install deps + dev tools
uv run uvicorn main:app --reload --port 8000     # http://localhost:8000
```

No key is needed: incidents run on the labelled offline agent. For Gemini, copy
`.env.example` to `.env` and set `GOOGLE_API_KEY`. Every setting is in
[docs/configuration.md](../docs/configuration.md).

`pyproject.toml` + `uv.lock` are the source of truth. `requirements.txt` is generated
from the lock for non-uv setups (`pip install -r requirements.txt`).

## Where to start reading

1. `src/loop.py`: `IncidentRun.run_loop()` is the whole loop on one screen.
2. `src/agents/`: one file per agent, each one model call returning a typed object.
3. `src/guardrails/manager.py`: everything the model is not allowed to decide.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | status, `llm_mode` (`gemini` or `offline`), model, engine |
| GET | `/api/config` | the limits and safety settings actually in force |
| GET | `/api/scenarios` | the fault catalog |
| POST | `/api/incidents` | start an incident `{scenario_id, service?, auto_approve?}` (operator/admin) |
| GET | `/api/incidents` | list incidents |
| GET | `/api/incidents/stats` | dashboard figures |
| GET | `/api/incidents/{id}` | full incident state and postmortem |
| GET | `/api/incidents/{id}/stream` | live events (Server-Sent Events, resumable with `Last-Event-ID`) |
| POST | `/api/incidents/{id}/approve` | answer the human gate `{action_id, decision: approve or reject, note?}` |
| POST | `/api/incidents/{id}/cancel` | stop the run before its next step |
| GET | `/api/incidents/{id}/usage` | tokens, cost and latency per agent |
| GET | `/api/memory`, `/api/memory/search?q=` | what the runbook memory holds, and what it would recall |
| POST | `/api/auth/login`, `/api/auth/logout`, GET `/api/auth/me` | optional login ([guide](../docs/persistence-and-login.md)) |

Interactive docs at <http://localhost:8000/docs> while the server runs.

## Tests and evaluation

```bash
uv run pytest -q                                   # no key, no network
IC_ENGINE=graph uv run pytest -q                   # same suite on the LangGraph engine
uv run ruff check .
uv run python -m evaluation.run_eval --stub        # offline evaluation
```

More in [docs/testing-and-evaluation.md](../docs/testing-and-evaluation.md).
