# CLAUDE.md - AI Incident Commander

Guidance for Claude Code (and any AI assistant) working in this project.

## What this is

An autonomous SRE / self-healing-ops system: a bounded, human-supervised **agent
loop** that diagnoses and remediates simulated production incidents. React console
+ FastAPI backend + Google Gemini (or the labelled offline agent when no key is set).

It is also a **teaching artifact**. Code here is read as much as it is run, so a
change that is correct but obscures the lesson is not an improvement. Comments
explain *why*, and several exist specifically to name a failure mode a reader
should recognize.

## The one rule

> **The model decides what to do. Code decides what is allowed.**

The LLM routes, diagnoses, proposes, critiques and verifies. It never sets a
ceiling, grants a permission, classifies its own risk, or waives an approval.

This is a security boundary, not a style preference: telemetry is
attacker-influenced text that flows into every prompt, so anything a model can
influence must not be able to widen the model's own permissions. **If violating a
rule would be a security incident, that rule belongs in code, never in a prompt.**

## Commands

```bash
# Backend (from backend/)
uv sync --extra dev
uv run pytest -q                                  # whole suite, no API key needed
uv run pytest tests/test_halt_conditions.py -q    # the ones that matter most
uv run ruff check .
uv run uvicorn main:app --reload --port 8000

IC_ENGINE=graph uv run pytest -q                  # the same suite on LangGraph (CI runs both)
uv run python -m evaluation.run_eval --stub       # deterministic eval, free
uv run python -m evaluation.run_eval --judge      # live, costs quota

# Frontend (from frontend/)
npm install && npm run dev
npm run lint && npm run test && npm run build

# Both
make help
```

## Layout

```
backend/src/
  loop.py              THE loop. IncidentManager (lifecycle, SSE, approvals) +
                       IncidentRun (one incident; start at run_loop()).
  graph.py             Same system as a LangGraph StateGraph (IC_ENGINE=graph).
  config.py            Every tunable. Grouped by concern.
  llm.py               Model seam: structured output, tiering, budgets, breaker.
                       get_llm() returns Gemini or the offline agent (IC_LLM_MODE).
  offline_llm.py       The offline agent: rule-based, reads telemetry, no key needed.
  agents/              7 agents, each a pure function. _prompting.py is shared.
  guardrails/manager.py  RBAC, risk, registry preconditions, recovery veto,
                       injection quarantine, redaction.
  memory/runbook.py    Episodic memory: BM25 retrieval over past incidents.
  models/              state.py (state machine) + schemas.py (agent contracts).
  sandbox/             Simulated stack + fault catalog.
  tools/               observability.py (read-only) + remediation.py (the one write).
backend/evaluation/    metrics.py explains WHY each metric exists. Read it first.
backend/migrations/    Alembic. Applied at startup (db/session.py _migrate). New model
                       field -> `alembic revision --autogenerate`, then read the file.
docs/adr/              Decisions that had a real alternative.
EXERCISES.md           17 guided experiments, several that break things on purpose.
```

## Invariants - do not break these

1. **Agents never mutate state.** `run(state, ..., llm) -> Schema`. The orchestrator
   writes. This is what lets both engines drive the same functions.
2. **Exactly one function mutates the world**: `tools/remediation.execute_remediation`,
   and only on an `APPROVED` action.
3. **Risk is assigned by `GuardrailManager`, never by the model.** The proposal's
   self-assessed `risk` field is deliberately overwritten.
4. **High-risk actions always require a human**, even with `auto_approve` on.
5. **Ground truth (`faulted`) is never in the telemetry bundle.** Leaking it makes
   every scenario trivial while still looking solved.
6. **Every terminal incident has a `HaltReason`** and a postmortem - including
   failures, cancellations and escalations. With no usable model the postmortem is
   built from the record (`postmortem.fallback`).
7. **Telemetry is sanitized before any agent sees it.** New tool output included.
8. **New actions go in the `ActionType` enum**, plus `ACTION_RISK` and every role's
   allowlist. A test asserts allowlists are subsets of the menu.
9. **Code has the last word on "resolved".** A Verifier `recovered=true` is refused
   while metrics exceed `IC_RECOVERY_MAX_*` (`GuardrailManager.recovery_not_confirmed`).
10. **Cancel is checked before every point of no return** (top of each pass, before
   the gate, before acting) in BOTH engines. The graph runs on a checkpointed copy,
   so a cancel must be written into the checkpoint (`push_cancel` in `run_graph`).
11. **Model-written text is fenced like telemetry** (triage summary, Critic
   objections, action results). Only facts from our own state are "trusted".
12. **Every human-gate outcome names its decider** (`RemediationAction.decided_by`:
   a username, `policy:auto_approve`, or `nobody:timeout`), plus `created_by` /
   `cancelled_by` on the incident. API endpoints pass the authenticated username.

## Conventions

- **Docstrings explain why.** Module docstrings carry the design reasoning; several
  name a specific bug the code prevents. Preserve that when editing.
- **Config over constants.** New behaviour that a learner might want to toggle
  belongs in `Settings` with an `IC_` prefix, so it can be A/B'd from the shell.
- **Ceilings are enforced in `loop.py`**, never requested in a prompt.
- **`PROMPT_VERSION`** in `agents/_prompting.py` is bumped whenever a system prompt
  changes meaning. It is recorded in every eval report.
- **`SCHEMA_VERSION`** in `models/schemas.py` is bumped on breaking contract
  changes; runbook memory ignores entries from older versions.
- Line length 120. Ruff (`E,F,W,I`). No type checker configured.

## Testing

Tests are organized by *concern*, not by module:

| File | Covers |
|---|---|
| `test_loop.py` | happy paths, run shape, agent participation |
| `test_halt_conditions.py` | every ceiling, driven to an actual stop |
| `test_guardrails.py` | RBAC, risk, injection quarantine, redaction |
| `test_evidence_loop.py` | `needs_more_data` → tools → re-diagnose |
| `test_memory.py` | retrieval quality, poisoning resistance |
| `test_agents.py` | prompt construction, routing policy, the Critic |
| `test_api.py` | HTTP: auth, rate limit, idempotency, SSE resume |
| `test_graph_parity.py` | both engines agree on every scenario |
| `test_llm.py` | pricing, budgets, circuit breaker, tiering |
| `test_evaluation.py` | the harness scores good and bad runs correctly |
| `test_auth.py` | password hashing, token signing/expiry, fail-closed roles |
| `test_regressions.py` | audited bugs, each driven on the real path in both engines |

Rules when adding tests:

- **Never call a real model.** Inject `FakeLLMClient` (the offline agent with no
  delay) via `IncidentManager(llm=...)`. `ScriptedLLMClient` overrides one schema;
  `FailingLLMClient` simulates an outage. `conftest.py` FORCES both key variables
  empty, so a key in your shell can never reach the suite.
- The stub **infers from telemetry text**, exactly like the real model. Do not make
  it look up answers by scenario id - that would test nothing.
- Test the *enforcement*, not the constant. Asserting `settings.max_iterations >= 1`
  proves nothing; drive an incident to the ceiling and assert it stopped.
- `conftest.py` pins the environment before `src` is imported and redirects runbook
  memory to a temp file. Settings are frozen dataclasses: override via the `env`
  fixture, not `monkeypatch.setattr`.
- Run the suite on BOTH engines (`IC_ENGINE=graph uv run pytest -q`); CI does.
  `test_graph_parity.py` alone was not enough: it drives the graph through
  `graph.run_incident`, not the manager path users run, and missed real drift.
  Tests marked `loop_engine` assert loop-specific orchestration and are skipped
  under `IC_ENGINE=graph`.
- Poll for `is_finalized()` (i.e. `ended_at`), not just a terminal status. Status is
  set the moment the loop stops; the postmortem lands a moment later.

## Traps worth knowing

- **The graph checkpointer serializes state.** `snapshot.values["incident"]` is a
  *copy*. Write decisions back with `update_state`, and merge results back with
  `_merge_graph_state` - a blanket `__dict__.update` overwrites the event log and
  erases the audit trail. Both were real bugs.
- **`token_usage` events carry cumulative totals.** The frontend store assigns them;
  summing multiplies the displayed cost by the number of emits.
- **SSE uses the default event name.** Named events would require the client to
  `addEventListener` per type, so every new event type would silently break the UI.
  The type travels in the payload.
- **Cost is checked twice**: at the top of each pass *and* pre-flight per model
  call. Only the second prevents a single iteration overshooting the ceiling.
- **After a verification, the loop must re-diagnose.** Proposing from a stale
  diagnosis is how a compound fault gets the same fix applied twice.

## When adding a feature

Ask, in order:

1. Does it belong in the model's decision space, or in code? (See The One Rule.)
2. Does it need a new halt condition or bound? Unbounded anything is a bug.
3. Can a learner turn it off and measure the difference? If so, make it a setting.
4. Does it need a scenario to be testable? A guardrail with no adversarial scenario
   is untested by construction.
5. Does it change a prompt's meaning? Bump `PROMPT_VERSION`.
6. Does it belong in an ADR? Only if there was a real alternative with a real cost.

## Honest limitations

Documented in `docs/architecture.md` §10, and worth keeping honest rather than
quietly fixing the docs:

- In-memory persistence with a retention cap; no durability across restarts in the
  explicit-loop engine.
- The sandbox is not real infrastructure.
- Rate limiting is in-process, so it does not hold behind multiple replicas.
- Injection defence is partial - the architecture, not the regex list, is the
  protection.
- The LLM judge measures plausibility, which correlates with quality without being it.
