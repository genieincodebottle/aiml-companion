# Interview guide

The questions this design attracts, with the answers the code supports. Each answer
points at something you can open and show. Back to the [README](../README.md).

**Q: Walk me through your agent loop and its halt conditions.**

It is a bounded control loop: observe read-only telemetry, route to a specialist,
diagnose, propose one action, adversarially review it, gate on a human for risky
actions, act through the single mutating call, verify from fresh telemetry, where code
can refuse a "recovered" verdict the metrics contradict. Six
hard halts: recovery, iteration ceiling, cost ceiling, wall-clock ceiling, human
rejection, approval timeout - plus operator cancel, and two soft halts that
escalate on persistent low confidence or repeated Critic vetoes. Every terminal
incident carries a typed `HaltReason`, so the eval measures *how* runs end, not
just whether they succeeded.

**Q: How do you stop an agent taking a dangerous action?**

Four independent layers, none reachable from model output. The action is a closed
enum, so it cannot invent one. It must be in the role's allowlist, and an
unrecognized role gets no permissions. Risk is classified by the guardrail layer,
never by the model - the proposal's self-assessed risk is deliberately overwritten,
because an agent that could grade its own rollback "low risk" could route around
the gate. Each action also carries preconditions checked in code against fresh telemetry
("no failover without a healthy alternate"). And high-risk actions always stop for a
human, even in auto-approve mode.

**Q: How do you defend against prompt injection from tool output?**

Redact secrets, neutralize injection-shaped imperatives leaving an audit marker,
then fence the remainder as untrusted data. But I would not claim that solves it -
a regex list raises the cost of easy attacks and nothing more. The real protection
is architectural: even a fully hijacked model can only emit an action from a closed
menu, within its role's allowlist, with high-risk items gated. The
`poisoned_telemetry` scenario tests exactly that end to end.

**Q: What happens when the agent is not sure?**

It says so, and that is wired to behaviour rather than displayed. `needs_more_data`
plus a list of requested tools makes the loop run those read-only tools and
re-diagnose, bounded by `IC_MAX_EVIDENCE_ROUNDS`. If confidence stays below the
floor, it escalates instead of acting. `silent_config_drift` is unsolvable without
that branch, and `unfixable_ca_expiry` is only passed by escalating.

**Q: How do you evaluate an agent like this?**

Four families of metric. Outcome, scored against each scenario's *expected*
outcome, so correct escalation counts as success. Efficiency. Safety, including
harmful-action rate and injection resistance. Process quality, via deterministic
trajectory checks and an LLM judge that is deliberately not told the outcome. Every
scenario runs N times, and I report stability and confidence intervals, because one
run of a stochastic system is an anecdote.

**Q: Why both a hand-rolled loop and LangGraph?**

The explicit loop is readable - every halt condition and the human `await` on one
screen. The graph gives durability and resumability: it checkpoints before the only
mutating node, so with a persistent checkpointer (it ships with the in-process
`MemorySaver`) a pending approval can survive a restart, which the explicit loop
cannot do. Both drive the same agents and the same guardrail layer, a parity
test asserts they agree on every scenario, and CI runs the whole suite on both
engines. That last part matters: an earlier version only ran the parity file, and
the graph engine quietly drifted (it ignored cancel and halted one pass early)
while everything stayed green. What the graph does *not* buy is safety - the
guardrails are the same code called from the same places.

**Q: Could you point this at real production infrastructure?**

Not as it stands, and the reason is the interesting part. The sandbox returns
`effective: True` the instant an action is applied, because it owns the ground
truth. That one field is load-bearing for the Verifier, the loop-back decision and
the accuracy number. On real infrastructure it becomes a statistical judgement over
a noisy signal with a propagation delay, and you have to separate "my fix worked"
from "traffic dropped" or "someone else fixed it". Writing a Prometheus adapter is
a day; solving that is not, and I have not solved it.

So I would ship it in tiers. Tier 0 is a read-only advisor - real telemetry in,
diagnosis and one proposed action with a falsifiable prediction out, into Slack.
No write access, most of the value, and only the read protocol changes. Tier 1 has
a human execute the proposal, which produces the labelled dataset that does not
exist yet. Tier 2 grants autonomy for one reversible, well-verified action class,
unlocked by the measured rate for that class rather than by a flag. The eval
harness stops being a teaching artifact at that point and becomes the gate.

**Q: What would you improve next?**

Leasing per target, so two incidents - or an agent and a human - cannot act on one
service at once. That is a genuine gap and a prerequisite for any write access.
Then a shared store for rate limiting, since in-process limiting does not hold
behind multiple replicas, and the approval registry in Redis so the gate survives a
restart in either engine. Then OpenTelemetry; the per-agent cost accounting is the
groundwork. All of it is listed in `docs/architecture.md` under what is still
missing, and in [ADR-0010](adr/0010-integrating-real-infrastructure.md).
