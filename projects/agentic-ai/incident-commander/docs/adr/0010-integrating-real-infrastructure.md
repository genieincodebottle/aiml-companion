# 0010. Integrate real infrastructure behind protocols, and earn autonomy in tiers

**Status:** Accepted (design); partially implemented

## Context

Everything the agents perceive comes from `SandboxStack`, and the one mutating
call is `execute_remediation(stack, action)`. Both are typed to that concrete
class. The obvious reading is that integrating a real system means writing a
Prometheus adapter and a Kubernetes adapter, and that this is mostly a plumbing
exercise.

It is not, and saying so plainly is the point of this record.

Look at what the sandbox returns from an action:

```python
{"applied": True, "effective": True, "progress": True, "reason": "..."}
```

That is **ground truth, returned instantly**. Real infrastructure never answers
that question, and certainly not synchronously. `effective` is doing more
load-bearing work than the entire fault catalog: the Verifier reads it, the
loop-back decision depends on it, and the evaluation's correct-outcome rate
inherits its certainty. Replace the sandbox with a real cluster and that one field
becomes a statistical judgement over a noisy signal with a propagation delay.

Three further gaps only appear once actions are real:

1. **Actions half-apply.** `apply_action` is atomic and total. A scale-out that
   brings up three of five pods leaves a state that is neither before nor after.
   There is no notion of partial application or reconciliation.
2. **Undo frequently does not exist.** You cannot un-restart a process. The action
   model has no reversibility contract, so the Critic and the human gate are
   reasoning about risk without knowing what is recoverable.
3. **Nothing owns a target.** Two incidents on the same service, or an agent and a
   human acting simultaneously, will interleave writes.

There is also a tension the current design cannot resolve as written. `ActionType`
is a closed enum of six ([ADR-0003](0003-closed-enums-over-free-text.md)), which is
what makes "hallucinate a new action" a decoding error rather than an outage. But a
real organisation's runbook is dozens of actions and specific to that
organisation. The safety property and the integration requirement pull in opposite
directions.

## Options

1. **Open the action type to strings for integrators.** Restores flexibility and
   destroys the property that ADR-0003 exists to protect. Rejected outright: the
   closed menu is the last line of defence when a prompt-injected model is choosing
   what to run.
2. **Ship adapters for a fixed set of popular backends** (Prometheus, Datadog,
   Kubernetes). Useful demo, wrong shape - it makes this project the integration
   point for every vendor, and still leaves the verification problem untouched.
3. **Two protocols plus a data-driven action registry, and treat autonomy as
   something earned per action class rather than switched on.**

## Decision

Option 3, in two parts.

### Part A - the seams

Two `Protocol`s, structural rather than inheritance-based so an integrator writes a
plain class and never imports a base:

- **`TelemetrySource`** - the read side. Everything the agents perceive.
- **`ActionExecutor`** - the write side. The only thing that changes the world.

`SandboxStack` becomes the reference implementation of both rather than the
hardcoded type. `TOOL_REGISTRY` and `execute_remediation` are retyped to the
protocols.

The action menu inverts. Instead of an enum with satellite `ACTION_RISK` and
`ROLE_ALLOWED` tables, each action **declares its own contract**:

| Field | Why it exists |
|---|---|
| `risk` | Classified outside the model, as today |
| `reversible` / `undo` | The Critic and the human gate need to know what is recoverable |
| `preconditions` | "Do not fail over without a healthy alternate" becomes checkable, not advisory |
| `verification_signal` | The falsifiable prediction, declared by the action rather than invented per incident |
| `settle_seconds` | How long before the signal means anything |
| `blast_radius` | What else this can affect |
| `required_role` | Feeds the existing RBAC allowlist |

The enum is then **generated from the registry at startup**. The model still faces
a closed menu and out-of-menu output is still a decoding error; the menu is now
org-specific and data-driven. ADR-0003's property is preserved by construction.

### Part B - autonomy is earned, not configured

Integration ships as three tiers, and the default is the lowest one:

| Tier | Behaviour | Write access |
|---|---|---|
| **0 - Advisor** | Real telemetry in; diagnosis, one proposed action and a falsifiable prediction out, to Slack or a ticket | none |
| **1 - Human-executed** | The agent proposes, a human executes via the existing approval gate | none |
| **2 - Narrow autonomy** | One reversible, well-verified action class, on a scoped set of targets | scoped allowlist |

Tier 2 is unlocked **per action class, on evidence**: the measured correct-action
rate for that class, on that estate, from Tier 1 data. Not by a global flag.

## Consequences

**Good**

- The loop does not change. Halt conditions, the human gate, the guardrail layer,
  cost ceilings and memory are already domain-agnostic; that is the part that
  transfers, and this decision keeps it untouched.
- The closed-menu safety property survives integration instead of being traded away
  for it.
- Preconditions and verification signals move from prose in a prompt into data the
  code can enforce. `failover` requiring a healthy alternate stops being a sentence
  the model may ignore.
- Tier 0 is genuinely deployable. It is most of the value at a small fraction of
  the risk, and it needs only the read protocol.
- The evaluation harness acquires a job beyond teaching: it becomes the gate that
  decides when an action class graduates.

**Bad**

- Two ways to describe an action exist during the migration - the enum and the
  registry - until the enum is fully generated. That is real duplication and a real
  chance to drift.
- Declaring a contract per action is more work than adding an enum member. That is
  the same deliberate friction ADR-0003 accepted, extended.
- Tiering is a process commitment, not just code. A team that ignores it and jumps
  to Tier 2 gets no protection from this design, and nothing here can stop them.

**Honest limits of this decision**

It does **not** solve verification. It creates the place where a real answer must
go (`settle_seconds`, `verification_signal`, noise bands) and stops the sandbox's
instant `effective: true` from being mistaken for one. Deciding whether a metric
moved because of your action or because traffic dropped is a statistics problem,
and this project does not currently solve it.

It also does not address partial application. `ActionExecutor` returns a result
that can express "partially applied", but reconciling a half-applied change is
backend-specific work an integrator must do.

## See also

- [`../integration.md`](../integration.md) - the practical guide
- [ADR-0002](0002-llm-routes-code-governs.md) - why the menu is closed at all
- [ADR-0003](0003-closed-enums-over-free-text.md) - the property this preserves
- [ADR-0004](0004-falsifiable-predictions.md) - the prediction that verification tests
