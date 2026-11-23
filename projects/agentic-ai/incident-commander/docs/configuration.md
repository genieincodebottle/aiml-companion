# Configuration

Every setting is an environment variable. Put them in `backend/.env` (copy
`backend/.env.example`, which documents each one), or export them in your shell.
Restart the backend after a change.

The values the running backend is actually using are served at
`GET http://localhost:8000/api/config`, so you never have to trust this page.

Back to the [README](../README.md).

## The model

| Variable | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | empty | Gemini key (free at <https://aistudio.google.com/apikey>). `GEMINI_API_KEY` also works. |
| `IC_LLM_MODE` | `auto` | `auto`: Gemini if a key is set, else offline. `gemini`: Gemini only, refuse without a key. `offline`: always the offline agent. |
| `IC_OFFLINE_STEP_DELAY_S` | `0.8` | Pause per offline agent call, so the console is watchable. `0` for instant. |
| `IC_MODEL_NAME` | `gemini-flash-latest` | The strong model: Diagnostician, Remediation, Critic. |
| `IC_MODEL_FAST` | `gemini-flash-lite-latest` | The cheap model: Triage, Verifier, Postmortem, Supervisor. `""` runs everything on the strong model. |
| `IC_MODEL_TIERING` | `true` | Use the two-model split above. |
| `IC_TEMPERATURE` | `0.2` | Model temperature. |
| `IC_LLM_TIMEOUT_S` | `60` | Per-call timeout. |
| `IC_LLM_MAX_RETRIES` | `2` | Retries per call, with jittered backoff. |
| `IC_LLM_BREAKER_THRESHOLD` | `5` | Consecutive failures before the circuit breaker opens for the process. |
| `IC_LLM_BREAKER_COOLDOWN_S` | `30` | How long the breaker stays open. |

The `-latest` aliases keep working when Google retires a pinned version. Pin a
specific version when you need repeatable measurements.

## The loop's limits (halt conditions)

| Variable | Default | What it does |
|---|---|---|
| `IC_MAX_ITERATIONS` | `6` | Passes through the loop before escalating. |
| `IC_MAX_COST_USD` | `0.50` | Cost ceiling per incident, checked before every model call. |
| `IC_MAX_WALLCLOCK_S` | `900` | Time ceiling per incident. |
| `IC_MAX_EVIDENCE_ROUNDS` | `2` | Extra telemetry the Diagnostician may request. `0` makes Silent Config Drift unsolvable. |
| `IC_MIN_ACTION_CONFIDENCE` | `0.6` | Below this, the agent gathers evidence or escalates instead of acting. |
| `IC_APPROVAL_TIMEOUT_S` | `300` | Seconds to wait at the human gate before escalating. |

## Safety

| Variable | Default | What it does |
|---|---|---|
| `IC_AGENT_ROLE` | `sre-bot` | What the agent may run: `sre-bot`, `sre-bot-restricted`, `read-only`. An unknown role may run nothing. |
| `IC_AUTO_APPROVE` | `false` | Skip the human gate for low and medium risk. High risk always needs a human. The console also has a per-run switch. |
| `IC_CRITIC_ENABLED` | `true` | Run the Critic before the human gate. |
| `IC_MAX_CRITIC_VETOES` | `2` | Vetoes allowed before escalating. |
| `IC_SUPERVISOR_ENABLED` | `false` | Route with the LLM Supervisor instead of the fixed policy (one extra call per pass). |
| `IC_QUARANTINE_INJECTIONS` | `true` | Remove injection-shaped telemetry text, not just flag it. |
| `IC_REDACT_SENSITIVE` | `true` | Redact secrets and personal data before any prompt. |
| `IC_RECOVERY_MAX_ERROR_RATE` | `0.02` | Code refuses "recovered" while the error rate is above this. |
| `IC_RECOVERY_MAX_P99_MS` | `1000` | ...or while p99 latency is above this. |

## Memory

| Variable | Default | What it does |
|---|---|---|
| `IC_MEMORY_ENABLED` | `true` | Recall similar past incidents, and store each finished one. |
| `IC_MEMORY_PATH` | `backend/data/runbook_memory.json` | Where episodes are stored (gitignored). |
| `IC_MEMORY_TOP_K` | `3` | Past incidents recalled per run. |
| `IC_MEMORY_MAX_ENTRIES` | `200` | Oldest episodes are dropped beyond this. |

## Engine

| Variable | Default | What it does |
|---|---|---|
| `IC_ENGINE` | `loop` | `loop`: the explicit loop. `graph`: the same agents as a LangGraph `StateGraph` with checkpoints. |

## Server

| Variable | Default | What it does |
|---|---|---|
| `IC_HOST` | `127.0.0.1` | Bind address. Loopback by default: with login off, anyone who can reach the port can approve actions. |
| `IC_PORT` | `8000` | Port when started with `python main.py`. |
| `IC_LOG_LEVEL` | `INFO` | Log level. |
| `IC_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated origins allowed to call the API from a browser. |
| `IC_API_KEY` | empty | If set, mutating endpoints require this in an `X-API-Key` header. |
| `IC_RATE_LIMIT_PER_MIN` | `60` | Per-client limit on mutating endpoints. `0` disables. |
| `IC_MAX_RETAINED_INCIDENTS` | `50` | Finished incidents kept in memory before the oldest are dropped. |

## Database and login (optional)

| Variable | Default | What it does |
|---|---|---|
| `IC_DATABASE_URL` | empty | Postgres URL. Empty means in-memory, which is fully supported. |
| `IC_DB_POOL_SIZE` / `IC_DB_MAX_OVERFLOW` | `5` / `10` | Connection pool sizing. |
| `IC_AUTH_REQUIRED` | `false` | Put the API and console behind a login. Needs a database. |
| `IC_JWT_SECRET` | placeholder | Token signing key. At least 32 bytes; the placeholder is refused. |
| `IC_JWT_TTL_MINUTES` | `720` | Login lifetime. |
| `IC_SEED_ADMIN_USER` | `admin` | First admin, created only in an empty users table. |
| `IC_SEED_ADMIN_PASSWORD` | empty | Empty: a random password is generated and printed once in the log. |

Details in [persistence-and-login.md](persistence-and-login.md).

## The console (frontend)

| Variable | Default | What it does |
|---|---|---|
| `VITE_API_URL` | empty | Backend base URL. Empty uses the dev proxy (`/api` to port 8000). |
| `VITE_API_KEY` | empty | Sent as `X-API-Key`. Compiled into the public bundle, so it is **not** a secret. |
| `IC_BACKEND_URL` | `http://localhost:8000` | Where the dev server proxies `/api` (shell variable for `npm run dev`). |
