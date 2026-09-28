# 0005. Enforce budgets at the tool boundary, never in the prompt

**Status:** Accepted

## Context

An agent loop can run away: retry the same edit forever, re-read the same file,
burn tokens on a repo it cannot fix. Something has to stop it, and there are two
fundamentally different places to put that something.

## Options

1. **Ask in the prompt.** "You have at most 20 tool calls; stop when you run out."
   Free, and unenforced - it is a request to a system that is not obliged to comply
   and cannot reliably count.
2. **Wrap the model client** and count calls there. Catches token spend, misses
   tool loops that do not go through the model.
3. **Meter at the tool registry**, where every tool call already funnels through
   one place.

## Decision

Option 3. `harness/base.py` defines `Budget` with hard ceilings on steps, tool
calls and cost; `tools/registry.py` calls `charge_tool()` on every dispatch and
`charge_step()` per loop iteration. Exceeding any ceiling raises `BudgetExceeded`,
which the orchestrator catches and turns into a terminated run with a reason.

The prompt is not told about the budget as a rule to obey. The budget is a
property of the environment the agent is running in.

## Consequences

**Good**

- The ceiling holds regardless of what the model does, including if it is
  jailbroken, confused, or replaced with a different provider. This is the same
  principle as incident-commander's "the LLM routes, code governs".
- One place to audit. Every tool call is metered, traced and hook-filtered at the
  same choke point, so there is no path that spends without being counted.
- It works identically in stub mode, which is why the offline eval can meaningfully
  report `mean_steps`.

**Bad**

- A hard stop mid-migration leaves a worktree in a partial state. The run is
  correctly marked failed, but the partial diff still exists and a human has to
  decide what to do with it.
- The ceiling is per repo, so a fleet of 50 repos has no global cap - one bad
  rulebook could burn 50 budgets. A fleet-level ceiling is the obvious next thing
  and is not implemented.
- Cost metering depends on the provider reporting usage. The Gemini worker computes
  it from reported tokens; the SDK worker reads `total_cost_usd`. A provider that
  reports neither would meter zero and the cost ceiling would silently never fire.
