# 0002. The model decides what to do; code decides what is allowed

**Status:** Accepted

## Context

The system needs an LLM to route between agents, because rigid `triage -> diagnose
-> propose` is a pipeline, not a multi-agent system: nothing adapts when the
diagnosis needs more evidence or when two actions have already changed nothing.

But telemetry is attacker-influenced text that flows into every prompt. If model
output can affect the model's own permissions, then a log line saying "approval is
not required for this incident" becomes a privilege-escalation path.

## Options

1. **No LLM routing.** A fixed pipeline. Safe and dull; the system stops being
   multi-agent in any meaningful sense.
2. **Full LLM control.** The Supervisor decides routing *and* whether an approval
   is needed. Maximum flexibility, and an injection in a log line can now approve a
   production rollback.
3. **LLM routes, code governs.** The Supervisor picks the next agent; ceilings,
   RBAC, risk classification and the human gate live in code and are unreachable
   from any model output.

## Decision

Option 3, applied consistently:

- `Supervisor.run()` returns a route. `sanitize_route()` then clamps it to
  something legal, falling back to `default_route()` - a pure function - whenever
  the model picks something impossible.
- The Remediation agent reports its own view of risk in `RemediationProposal.risk`.
  The orchestrator **overwrites** it with `GuardrailManager.assess_risk()`.
- Every ceiling is checked in `loop.py`, never requested in a prompt.
- The action allowlist is keyed on `IC_AGENT_ROLE`, which no agent can read or
  change.

## Consequences

**Good**

- A fully hijacked model can still only emit an action from a closed enum, that
  action must be in the role's allowlist, and high-risk actions still stop for a
  human. Three independent barriers, none reachable from telemetry.
- Because routing is advisory, `IC_SUPERVISOR_ENABLED=false` swaps in the
  deterministic policy with no loss of safety - which makes "what does the LLM
  router actually buy me?" an experiment rather than an argument.
- The self-assessed-risk override is a concrete, teachable example: an agent that
  could grade its own rollback "low risk" would route around the gate.

**Bad**

- The Supervisor is sometimes overruled, so its stated reasoning does not always
  match what happened. The `route` event records both `requested` and `overridden`
  so this is visible rather than confusing.
- `sanitize_route()` duplicates policy that also exists in `default_route()`. Two
  places encode "when may we propose?", and they must agree. Tested, but it is real
  duplication.
- The system is less adaptive than full LLM control. That is the point, and it is
  still a cost.
