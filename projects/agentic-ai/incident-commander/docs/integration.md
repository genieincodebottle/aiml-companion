# Integrating your own system

How to point this loop at real telemetry and real actions, and — more importantly —
in what order, and what will actually be hard.

If you read nothing else: **the adapters are the easy part.** A Prometheus reader is
a day's work. What is hard is answering *"did it work?"* about a real system, and
that problem is not solved here. See [What this does not solve](#what-this-does-not-solve).

---

## Start here: what transfers, and what does not

| Transfers unchanged | Must be replaced |
|---|---|
| The loop, its six halt conditions and two soft halts | `SandboxStack` — telemetry |
| The human approval gate and the approval timeout | `SandboxStack` — actions |
| Guardrails: RBAC, risk classification, repeat-action block | The action menu (yours will not be these six) |
| Injection quarantine and secret redaction | The fault catalog (yours is your incident history) |
| Cost ceilings and per-agent accounting | |
| Episodic memory and its poisoning mitigations | |
| The evaluation harness | |

Nothing in `loop.py` is sandbox-specific. That is the property worth relying on.

---

## Do this in tiers, not in one step

The single strongest recommendation in this document.

| Tier | What it does | Write access | Prerequisite |
|---|---|---|---|
| **0 — Advisor** | Real telemetry in; diagnosis, one proposed action and a falsifiable prediction out, to Slack or a ticket | **none** | a `TelemetrySource` |
| **1 — Human-executed** | The agent proposes, a human executes through the existing approval gate | none | + your runbook in the registry |
| **2 — Narrow autonomy** | One reversible, well-verified action class on a scoped set of targets | scoped allowlist | evidence from Tier 1 |

**Tier 0 is most of the value at a small fraction of the risk**, and it needs only
the read protocol. Deploy it, and you have a system that reads an alert, gathers
evidence, forms a ranked diagnosis with a confidence number, proposes one action
and predicts what should change — into a Slack thread, where a human decides.

**Tier 1 generates the data you do not have.** Every proposal a human accepts or
overrides is a labelled example on *your* estate. That is the dataset that tells
you whether Tier 2 is justified, and it does not exist until you run Tier 1.

**Tier 2 is unlocked per action class, on measured evidence** — not by a global
flag. This project reports roughly 80% correct-outcome on a deliberately toy fault
catalog. Nobody should grant production write access on the strength of that. The
evaluation harness is not only a teaching artifact; it is the gate.

---

## Step 1 — a `TelemetrySource` (Tier 0)

Three required methods. No base class to inherit; a plain object that has them
conforms.

```python
from typing import Any

class PrometheusSource:
    """Reads a real service. MUST be side-effect free."""

    def __init__(self, client, service: str):
        self._c, self._service = client, service

    def metrics(self) -> dict[str, Any]:
        return {
            "p99_latency_ms": self._c.query(f'histogram_quantile(0.99, ...{self._service}...)'),
            "error_rate": self._c.query(f'rate(http_errors_total{{service="{self._service}"}}[5m])'),
            "cpu": self._c.query(f'...'),
        }

    def logs(self, limit: int = 20) -> list[str]:
        return self._c.loki_tail(self._service, limit)   # UNTRUSTED text

    def traces(self) -> dict[str, Any]:
        return self._c.tempo_summary(self._service)
```

Check it conforms:

```python
from src.integrations import TelemetrySource
assert isinstance(PrometheusSource(c, "checkout"), TelemetrySource)
```

Three rules that are not style preferences:

1. **Side-effect free.** The read/write split is what lets diagnosis run unguarded
   while every action passes a gate. If reading could change production, the
   guardrail layer is protecting nothing.
2. **Never return a ground-truth health flag.** Agents must infer health from the
   numbers. Leaking a boolean answer makes every incident trivial while still
   looking like reasoning — a test asserts this for the sandbox
   (`test_metrics_never_leak_ground_truth`).
3. **Treat logs and traces as hostile.** They contain user- and attacker-influenced
   text. The guardrail layer sanitizes them before they reach a prompt; your
   adapter must not pre-trust them.

Optional methods (`pool_stats`, `config_diff`, `dependency_health`, `slo_burn`)
back the evidence-gathering loop. Raise `NotImplementedError` for ones you lack —
`run_tool` turns that into an error dict rather than killing the loop, which is the
right degradation when a telemetry backend is down mid-incident.

---

## Step 2 — declare your actions (Tier 1)

Your runbook is not `restart_service` and five friends. Declare each action in
`DEFAULT_ACTIONS` in [`src/integrations/registry.py`](../backend/src/integrations/registry.py),
**and** add its name to the `ActionType` enum in `models/state.py` (plus its risk in
`guardrails/manager.py` and the role allowlists). The enum is still what the model
chooses from; `verify_matches_enum()` fails the tests if the two drift apart.

```python
ActionSpec(
    name="drain_node",
    description="Cordon and drain a node, rescheduling its pods.",
    risk=ActionRisk.HIGH,
    reversible=True,                       # uncordon
    verification_signal="pod evictions complete and error rate returns to baseline",
    settle_seconds=240,                    # eviction + reschedule + readiness
    blast_radius="every pod on the node; capacity drops until they reschedule",
    preconditions=(_requires_spare_capacity,),
    allowed_roles=frozenset({"sre-bot"}),
)
```

The model still chooses from a **closed** menu - out-of-menu output stays a
decoding error ([ADR-0003](adr/0003-closed-enums-over-free-text.md)). Building the
menu itself from this registry (instead of the enum) is the remaining migration in
[ADR-0010](adr/0010-integrating-real-infrastructure.md).

Two fields do work nothing in the original design did:

**`preconditions`** turn a prompt instruction into an enforced check, and they are
enforced today: `GuardrailManager.check_action` runs them on every proposal, in both
engines, against telemetry the loop fetches itself. "Do not fail
over without a healthy alternate" was previously only a sentence — and a live run
showed why that is not enough: the model proposed failover with no evidence an
alternate existed, because nothing could stop it. A precondition returns a *reason
string*, not a bool, because that string goes to the operator and into the
postmortem. A precondition that raises is treated as a refusal, never as a pass.

**`settle_seconds`** is the admission that verification takes time. The sandbox
answers instantly because it owns the truth. An agent that reads a metric one
second after a restart is reading noise.

---

## Step 3 — an `ActionExecutor` (Tier 2)

```python
from src.integrations import ActionResult

class K8sExecutor:
    def supports(self, action_type: str) -> bool:
        return action_type in {"restart_service", "scale_out", "rollback_deploy"}

    def apply_action(self, action_type: str, **kw) -> dict:
        if action_type == "scale_out":
            before = self._replicas()
            self._scale(before + 2)
            ready = self._wait_ready(timeout=120)
            return ActionResult(
                applied=ready == before + 2,
                partial=before < ready < before + 2,     # the awkward middle
                effective=None,                          # too early to say
                reason=f"{ready}/{before + 2} replicas ready",
                undo_token=str(before),
            ).to_dict()
```

Four rules:

1. **Never raise for an ordinary refusal.** Return `applied=False` with a reason.
   The loop is built to read a refusal and re-diagnose; an exception turns "I would
   not do that" into a crash.
2. **Be idempotent per `action_id`.** A retried approval must not scale out twice.
3. **Return `effective=None` unless you genuinely know.** This is the default for a
   reason — an optimistic adapter is the easiest way to make an autonomous
   remediator dangerous.
4. **Set `partial=True` when the world moved but did not arrive.** Something must
   reconcile that state, and it cannot if you report a clean failure.

Implement `undo()` (the `ReversibleExecutor` protocol) only where a real undo path
exists, and set `reversible=True` only then. Most actions are not reversible: you
cannot un-restart a process.

---

## Step 4 — build your own evaluation

**Do not skip this, and do not trust the shipped numbers.** They describe a toy
catalog. Yours must be built from your own incident history:

- Replay real past incidents with their real telemetry.
- Include **negative cases** — incidents where the correct answer was to escalate
  or do nothing. An agent evaluated only on faults it can fix will invent a fix for
  one it cannot, and no outcome metric will catch it
  ([ADR-0006](adr/0006-negative-scenarios-in-eval.md)).
- Include **adversarial cases** — logs containing injection attempts.
- Track **harmful-action rate** as the primary safety metric. Correct-outcome rate
  is the headline; harmful-action rate is the one that should block a release.

Promote an action class to Tier 2 when its measured correct-action rate on *your*
estate justifies it. Not before.

---

## What this does not solve

Stated plainly, because a guide that claims completeness is not credible.

**Verification.** The largest gap. The sandbox returns `effective: True` instantly
because it owns the ground truth. In production, "did my action work?" is a
statistical question over a noisy signal with propagation delay, and you must
separate *my fix worked* from *traffic dropped* / *someone else fixed it* / *the
deploy freeze started*. `settle_seconds` and `verification_signal` create the place
where a real answer goes; they do not produce one. You will need noise bands, and
probably a control comparison.

**Partial application and reconciliation.** `ActionResult.partial` lets an adapter
*say* the change half-landed. Deciding what to do about it is backend-specific and
unimplemented here.

**Concurrency on a target.** Nothing prevents two incidents — or an agent and a
human — acting on one service simultaneously. You need leasing per target before
Tier 2. This is a genuine gap, not a subtlety.

**Change-management context.** Freeze windows, ticket linkage, on-call paging and
audit export are all absent.

**Multi-tenancy.** One agent role process-wide. Per-team estates need per-tenant
roles, budgets and memory isolation.

---

## A caution about the name

"Autonomous SRE" oversells the interesting part and undersells the useful one. What
this actually is — and what is worth integrating — is **a bounded, auditable
decision loop with a human gate**. The autonomy is the least interesting property
and by far the hardest to justify. The auditability, the halt conditions and the
falsifiable predictions are what a production estate actually benefits from.

Tier 0 delivers all three, and takes no write access at all.

---

## See also

- [ADR-0010](adr/0010-integrating-real-infrastructure.md) — why the seams are shaped this way
- [`src/integrations/protocols.py`](../backend/src/integrations/protocols.py) — the two protocols
- [`src/integrations/registry.py`](../backend/src/integrations/registry.py) — the action contract
- [`tests/test_integrations.py`](../backend/tests/test_integrations.py) — conformance and drift guards
