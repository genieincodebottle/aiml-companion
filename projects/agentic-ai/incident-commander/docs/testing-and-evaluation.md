# Testing and evaluation

Everything on this page runs **offline, for free**, unless it says "live".
Back to the [README](../README.md).

## Tests

```bash
cd backend
uv sync --extra dev
uv run pytest -q                       # the whole suite, no key, no network
IC_ENGINE=graph uv run pytest -q       # the same suite on the LangGraph engine
uv run ruff check .                    # lint

cd ../frontend
npm run lint && npm test && npm run build
```

(Windows PowerShell: `$env:IC_ENGINE = "graph"; uv run pytest -q`.)

### End-to-end, in a real browser

```bash
cd frontend
npx playwright install chromium     # once
npm run test:e2e
```

This starts its own backend (forced into offline mode, so no key is used even if
one is configured) and its own console on dedicated ports (8765, 5765), then drives
the journeys a first-time user takes: the offline banner on every page, trigger,
approve, reject, cancel at the gate, the postmortem naming who approved, and the
History page. A test fails on **any** browser console error or React warning, which
is what catches a library upgrade that silently breaks rendering. CI runs it on
every push.

The tests use the offline agent from `src/offline_llm.py`. It infers the fault from
the **telemetry text in the prompt**, the same input Gemini gets. It never looks up
answers by scenario id, so if a prompt stops carrying its evidence, tests fail.

Tests are organized by what they protect, not by module:

| File | Covers |
|---|---|
| `test_loop.py` | happy paths, the shape of a run, which agents take part |
| `test_halt_conditions.py` | every ceiling, driven to an actual stop |
| `test_guardrails.py` | roles, risk, injection quarantine, redaction |
| `test_evidence_loop.py` | "I need more data" -> tools -> re-diagnose |
| `test_memory.py` | retrieval quality, memory poisoning |
| `test_agents.py` | prompt construction, routing, the Critic |
| `test_api.py` | HTTP: offline mode, auth, rate limit, idempotent approvals, SSE resume |
| `test_graph_parity.py` | both engines agree on every scenario |
| `test_audit_trail.py` | who started, approved, rejected or cancelled, on both engines |
| `test_migrations.py` | the Alembic migrations create every table and index the models define |
| `test_regressions.py` | bugs found by audit (cancel, engine drift, outages, preconditions...), each driven on the real path |
| `test_llm.py` | pricing, budgets, circuit breaker, model tiers |
| `test_evaluation.py` | the harness scores good and bad runs correctly |
| `test_auth.py` | password hashing, tokens, fail-closed roles |
| `test_integrations.py` | adapter protocols, the action registry |

## Evaluation

```bash
cd backend
uv run python -m evaluation.run_eval --stub                     # offline, ~3 s
uv run python -m evaluation.run_eval --stub --check-baseline    # fail on regression (CI)
uv run python -m evaluation.run_eval --scenarios bad_deploy     # live: one incident
uv run python -m evaluation.run_eval --repeats 3                # live: all ten, three times
uv run python -m evaluation.run_eval --repeats 3 --judge        # live + LLM judge
```

**`--stub` measures the harness, not the agent.** The report says so. It proves the
loop and the scoring work; only a live run measures a model.

Each run starts from an **empty runbook memory**, so results do not drift between
runs. Pass `--persist-memory` to use the real memory file instead.

### What is measured

| Family | Metrics | Question it answers |
|---|---|---|
| Outcome | correct-outcome rate with a 95% interval | did it end the right way, for the right reason? |
| Efficiency | correct first action, iterations, cost | what did it cost to get there? |
| Safety | harmful-action rate, injection resisted | what did it do on the way? |
| Process | trajectory score, optional LLM judge | was the reasoning sound? |
| Reliability | stability across repeats | would it do the same again? |

Two details that make the numbers honest:

- **"Escalated" only counts as correct when the agent chose to escalate** (low
  confidence, the Verifier, the Critic, a guardrail). An escalation caused by an
  outage, a crash or an exhausted ceiling ends with the same status, but it is
  scored as wrong. Otherwise a dead model would score 100% on the "must escalate"
  scenario.
- **The judge is never told the outcome.** A judge that knows a run succeeded
  invents a justification for it.

[`evaluation/metrics.py`](../backend/evaluation/metrics.py) explains why each
metric exists.

## Cost and time

| What you run | Model calls | Time | Cost |
|---|---|---|---|
| Tests, offline eval, notebook | 0 | seconds | free |
| One live incident, one pass | 5-6 | ~30 s | ~$0.01 |
| One live incident, 2-3 passes | 10-16 | 60-90 s | ~$0.03 |
| Live eval, all ten scenarios | ~90 | 10-12 min | ~$0.22 |
| Hard ceiling per incident | - | - | $0.50 |

Measured on `gemini-2.5-flash` with the fast tier on and the Supervisor off; the
current `-latest` aliases may price differently. Two settings dominate cost:

| Setting | Default | If you flip it |
|---|---|---|
| `IC_MODEL_FAST` | `gemini-flash-lite-latest` | `""` (everything on the strong model): about +45% cost, +40% time |
| `IC_SUPERVISOR_ENABLED` | `false` | `true`: one more call per pass, about +20 s per incident |

The Supervisor is off by default because its answer is clamped to the same fixed
policy anyway. To see that for yourself (live), compare
`IC_SUPERVISOR_ENABLED=true` against the default on the same scenarios with
`--out` and diff the `routes` of each run. Finding that it changes nothing is a real
result.
