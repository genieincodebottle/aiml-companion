# Configuration

Every setting is an environment variable, read from `backend/.env` (local runs) or
the project-root `.env` (docker compose). A variable already set in your shell wins.
Everything is optional: with nothing set, the platform runs offline.

## Model

| Variable | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | unset | Turns on live mode. `GEMINI_API_KEY` is accepted too. Free key: https://aistudio.google.com/apikey |
| `MC_MODE` | `auto` | `auto`: live if a key is set. `offline`: never call a model. `live`: refuse to start cases (503) if there is no key |
| `MC_PREMIUM_MODEL` | `gemini-flash-latest` | Model for premium-tier agents (graph, narrative, critic, supervisor, the judge) |
| `MC_CHEAP_MODEL` | `gemini-flash-lite-latest` | Model for cheap-tier agents (triage, screening) |
| `MC_TEMPERATURE` | `0.2` | Sampling temperature |
| `MC_LLM_TIMEOUT_S` | `60` | Per-call timeout. The SDK retries once, then the step retries once, then the case dead-letters |
| `MC_FORCE_STUB` | unset | Older name for `MC_MODE=offline`; still honoured |

The `-latest` aliases are Google's stable pointers. Pinned versions get retired for
new accounts and then return 404.

## People

| Variable | Default | What it does |
|---|---|---|
| `MC_OPERATORS` | unset | `name:role+role:token,...`. Set it and every write needs a bearer token. Roles: `analyst` opens cases, `reviewer` decides them |
| `MC_SIGNING_KEY` | random per process | Signs agent workload tokens. Set it only if several processes must verify each other's tokens |

See [operators-and-approvals.md](operators-and-approvals.md).

## Gateway

| Variable | Default | What it does |
|---|---|---|
| `MC_CACHE_ENABLED` | `1` | Semantic cache, scoped to one case |
| `MC_CACHE_SIMILARITY` | `0.95` | Cosine similarity needed for a cache hit |
| `MC_REDACT_PII` | `1` | Mask emails, SSNs, IBANs, card and phone numbers before any provider sees them |

## Ceilings and the gate

| Variable | Default | What it does |
|---|---|---|
| `MC_MAX_COST_USD` | `0.25` | Model spend per case, checked before every call. A live case uses about $0.004 |
| `MC_MAX_WALLCLOCK_S` | `600` | Running time per case. Time parked at the gate does not count |
| `MC_MAX_STEPS` | `40` | Step ceiling |
| `MC_AUTO_APPROVE` | `0` | `1` satisfies the reviewer gate automatically (demos and the eval). Recorded as `policy:auto_approve` |
| `MC_DISABLE_POLICY` | `0` | Teaching flag only: runs the fleet with no policy engine. Never in production |
| `MC_EVAL_THRESHOLD` | `0.80` | Mean judge score the promotion gate needs |

## Tracing

| Variable | Default | What it does |
|---|---|---|
| `MC_OTEL_EXPORTER` | unset | `console` or `otlp` exports real OpenTelemetry spans. Needs `uv sync --extra otel`. `otlp` reads the standard `OTEL_EXPORTER_OTLP_*` variables |

Spans always go to `.work/traces/<workflow>.jsonl` as well.

## Server and storage

| Variable | Default | What it does |
|---|---|---|
| `MC_DATA_DIR` | `backend/.work` | Checkpoints, event logs, traces, the audit log, the DLQ and the prompt registry |
| `MC_HOST` / `MC_PORT` | `127.0.0.1` / `8000` | Used by `uvicorn` only if you pass them; the Quick Start passes `--port` |
| `MC_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated origins allowed to call the API from a browser |
| `MC_LOG_LEVEL` | `INFO` | Python log level |
