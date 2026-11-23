# 0009. No heuristic fallback when the model is unavailable

**Status:** Accepted, amended 2026-11-23 (see the end: a *labelled* offline mode replaces the hard 503)

## Context

A demo that requires an API key is a demo many people never run. The obvious fix is
a deterministic fallback: if there is no key, drive the loop with keyword rules so
the UI still animates.

Several well-known agent projects do exactly this. It is also how a project quietly
stops being an agentic system while continuing to describe itself as one.

## Options

1. **Heuristic fallback in the app.** Always runnable. The thing people evaluate is
   then frequently not the agent, and nobody can tell which mode produced a given
   result.
2. **Mandatory model, hard failure.** Honest, and raises the bar to running it.
3. **Mandatory model in the app, deterministic stub in tests only.**

## Decision

Option 3.

- `LLMClient.__init__` raises `LLMConfigError` when no usable key is configured.
- `POST /api/incidents` returns **503** with a message pointing at the free key,
  rather than starting an incident that would not be agentic.
- `/api/health` reports `degraded` so the UI can say so before anyone clicks.
- `tests/_fake_llm.py` provides `FakeLLMClient`, injected only through
  `IncidentManager(llm=...)`. It is never importable from the running app.
- The stub is written to *infer from telemetry text* rather than to look up answers
  by scenario id, so the tests still fail if a prompt stops carrying the evidence
  the real model needs.

## Consequences

**Good**

- What you see running is always the real system. There is no mode where the
  console looks agentic and is not.
- The failure is loud and actionable: a 503 naming the environment variable and the
  URL to get a free key beats a UI that silently does something simpler.
- Tests and the `--stub` evaluation still run offline, for free, deterministically,
  in CI.

**Bad**

- **A hard prerequisite.** Anyone without a key sees a 503, and some will not get
  past that. Mitigated by the free tier and by the README leading with it, not
  eliminated.
- The stub is a second implementation of "what a competent SRE would conclude", and
  it can drift from what real models do. It tests the *loop*, never agent quality -
  and the eval report says so explicitly in stub mode, because conflating the two
  would undo the honesty this decision exists to buy.
- CI can only verify control flow, not reasoning. Judging reasoning needs a live
  model and real quota, so it stays a manual `--judge` run.

## Amendment (2026-11-23): a labelled offline mode, never a silent one

**What changed.** The "hard prerequisite" consequence above turned out to be the
biggest reason people never ran the console at all. The decision is amended:

- With no key, `IC_LLM_MODE=auto` (the default) runs the agents on the
  deterministic rule-based agent, moved from `tests/_fake_llm.py` to
  `src/offline_llm.py` so the app can use it.
- It is **labelled everywhere**: `/api/health` reports `llm_mode: "offline"`, the
  console shows an "Offline mode" banner on every page, and every incident records
  `llm_mode`. There is no state in which a viewer can mistake it for model
  reasoning.
- It is **never a fallback after a failure**. If Gemini is selected and goes down
  mid-incident, the incident escalates (`llm_unavailable`); it does not switch
  brains. That was the actual failure mode this ADR guards against, and it still
  cannot happen.
- `IC_LLM_MODE=gemini` restores the original behaviour (503 without a key).

**What it still costs.** The offline agent is a second implementation of "what a
competent SRE would conclude" and can drift from real models. It demonstrates the
loop, never agent quality, and the eval report keeps saying so in `--stub` mode.

