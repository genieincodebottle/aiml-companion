# Architecture deep dive

Mission Control is organised around one decision: **separate the control plane from
the data plane.** The control plane holds configuration and policy: which agents
exist, what they may do, which prompt version is live and who may decide what. The
data plane executes: the gateway routes model calls and the runtime runs workflows.
Most of the rest follows from keeping those two apart.

## The request path

A tenant workflow never touches a model provider or a tool directly. Each step gets a
`WorkflowContext` and calls `call_model` or `call_tool`, which pass through the planes:

```
call_model(agent, user, schema)
  └─ Gateway.complete
       ├─ budget check                     over the case's ceiling -> dead-letter
       ├─ redact(system, user)             PII -> stable typed placeholders
       ├─ route(tier -> model)             the agent card says cheap or premium
       ├─ cache.lookup(user, key)          a near-duplicate prompt is served free
       ├─ provider[tier].complete(...)     offline brain or Gemini; on error, the other tier
       ├─ cost[workflow] += usage          per-case spend, under a lock
       └─ audit + cache.store              the audit record names the model that answered

call_tool(agent, tool, fn, **args)
  ├─ identity.mint(card) + verify(tool)    scoped, signed, short-lived token
  ├─ policy.evaluate(card, tool, args)     ALLOW / REQUIRE_APPROVAL / DENY
  ├─ fn(**args)                            the actual tool
  └─ scan_for_injection(result)            hostile instructions neutralised, event recorded
```

Because every call passes the same chokepoints, budgets, caching, PII masking, model
routing, authorisation and audit are platform guarantees, not things each agent
author has to remember.

An earlier version routed every live call to the premium model whatever tier the
agent asked for, while the audit log named the cheap one. One provider per tier, each
falling back to the other, makes the audit record true; `test_platform_claims.py`
checks it.

## Durable execution

The runtime is event-sourced. A workflow's history is an append-only event log
([`models/events.py`](../backend/src/models/events.py)) mirrored to JSONL. The engine
([`runtime/engine.py`](../backend/src/runtime/engine.py)) drives a workload's named
steps and writes a checkpoint after each one: state, completed steps, the pending
action and spend. The write goes to a temp file that is renamed over the old one, so
a crash never leaves a half-written checkpoint.

- **Resume after a restart.** A reviewer may decide hours later, after a deploy. If
  the case is not in memory, `_rehydrate` rebuilds it from the checkpoint and event
  log: the workload is recreated from a tenant registry, in-process objects such as
  tool clients are recreated by `Workload.rehydrate`, and the spend is reloaded so the
  cost ceiling still counts it. Completed steps are skipped, so nothing re-runs or is
  billed twice. An unknown checkpoint version is refused rather than guessed at.
- **Audit** comes free: the event log is the case history.
- **Debugging** is a diff of two event streams.

Earlier versions wrote checkpoints but never read them back for a write: deciding a
case after a restart raised "restart or replay required". `_rehydrate` closes that.

## The reviewer gate and maker-checker

Before any `side_effecting` step the engine asks the policy engine about the pending
action. `REQUIRE_APPROVAL` (every case disposition, and any case of $100k or more)
parks the workflow: status `awaiting_approval`, an `approval_requested` event, a
checkpoint, and control returns. The API ran the steps on a worker thread, so the
console watched all of this on the event stream.

A decision then goes through `resume_after_approval(decision, note, actor)`:

- **Maker-checker.** The person who opened the case may not decide it, whatever
  roles they hold. It is the four-eyes control every AML desk runs. A refused attempt
  is recorded as `maker_checker_blocked`.
- **One decision per case.** A per-case lock serialises decisions. Repeating the
  same decision is harmless (the API answers `duplicate: true`), and a different one
  is refused with who decided first.
- **The audit trail.** `started_by`, `decided_by`, `decided_at`, the decision and the
  reviewer's note are stored on the case and shown in the console. With
  `MC_AUTO_APPROVE=1` the record says `policy:auto_approve`, never a person's name.

Approving continues the loop from the gated step. Rejecting escalates the case to a
person and commits nothing. Who is acting comes from an operator token when
`MC_OPERATORS` is set, or from an unverified `X-Operator` header with login off; see
[operators-and-approvals.md](operators-and-approvals.md).

## Reliability: saga, ceilings, dead letters

The disposition commit records a case and notifies a downstream system, and that
pair is not atomic. The commit runs inside a `Saga`
([`runtime/saga.py`](../backend/src/runtime/saga.py)): each forward sub-action registers
a compensation, and a failure unwinds them in reverse. A notify failure removes the
case record rather than leaving a half-filed SAR, and the case ends `compensated` and
escalated. Try `uv run python cli.py run ALRT-001 --fault`.

Three ceilings stop a runaway case: spend (`MC_MAX_COST_USD`, checked before every
model call), wall-clock time spent running (`MC_MAX_WALLCLOCK_S`; time parked at the
gate does not count) and step count. A step that fails is retried once. A case that
hits a ceiling, fails twice, or crashes the runtime goes to the dead-letter queue
([`runtime/dlq.py`](../backend/src/runtime/dlq.py)) with its reason and last state. It is
parked, never silently dropped, and the reason is shown on the case.

## Security model

- **Least privilege.** A tool must be on the agent card's allowlist, or the policy
  engine denies it. No rule matching means deny.
- **Zero trust between agents.** An agent carries no ambient authority. To call a
  tool it presents a token minted for its card, scoped to that card's tools,
  HMAC-signed and valid for 300 seconds. The signing key comes from `MC_SIGNING_KEY`
  or is generated per process; it is never in the source
  ([`identity.py`](../backend/src/identity.py)).
- **Indirect prompt injection.** Tool output (customer notes, related-party notes)
  is attacker-influenceable text. It is scanned for hijack phrasing before it reaches
  a prompt and neutralised with an auditable event
  ([`guardrails.py`](../backend/src/guardrails.py)). ALRT-004 plants such a note on the
  path to a sanctioned shell. A pattern list does not solve injection; the closed tool
  lists and the reviewer gate are what make a hijacked agent harmless.
- **PII** is masked at the gateway boundary before any provider or audit sink sees it.
- **People.** Analysts open cases and reviewers decide them. The API binds to
  127.0.0.1 by default, CORS never falls back to `*`, and workflow ids are checked
  before they are used as file names.

## Evaluation and the promotion gate

[`evaluation/run_eval.py`](../backend/evaluation/run_eval.py) runs every golden alert
through the whole platform and scores each case three ways: does the disposition
match the ground truth, was a planted injection caught, and what does a judge score
for grounding and soundness. The judge scores structural facts computed by the eval
(`uncited_claims=`, `evidence_count=`), not the narrative's description of itself
([ADR 0007](adr/0007-a-judge-that-can-fail.md)).

The gate passes only if every case is right, every planted injection was caught and
the mean judge score clears `MC_EVAL_THRESHOLD`. With `--candidate <file>` it scores a
new prompt under the registry's `candidate` label and, only if a **live** run passes,
moves `production` to it and saves the registry. The server reads `production`, so
the next case uses the new prompt. Rollback is moving the pointer back
(`cli.py prompts rollback`). An offline run never promotes, because the offline brain
ignores prompt wording and cannot have measured the change
([ADR 0010](adr/0010-promote-only-on-a-live-measurement.md)).

> **How independent is the judge?** Live, it is a separate call to a model of the
> same family: a second opinion, and self-preference is documented for LLM judges.
> Offline, judge and subject come from the same deterministic brain, so it is a
> consistency check, not an opinion.

## Observability

One trace per workflow. Every span records `trace_id`, `span_id` and
`parent_span_id`: model and tool spans nest under the step that made them, and the
trace id is derived from the workflow id, so a case paused at the gate and resumed
after a restart is still one trace. Spans always go to
`.work/traces/<workflow>.jsonl`. With `MC_OTEL_EXPORTER=console` or `otlp` (and the
`otel` extra) they are also exported as real OpenTelemetry spans under the same trace
id, for Jaeger, Grafana Tempo, Honeycomb or Langfuse
([`observability.py`](../backend/src/observability.py)).

## Where the seams are

| In this repo | Swap for | Touch only |
|---|---|---|
| offline brain (`StubProvider`) | any model provider | `gateway/providers.py` |
| JSONL event log + checkpoints | Kafka + an event store | `models/events.py`, `runtime/engine.py` |
| in-process semantic cache | Redis + real embeddings | `gateway/cache.py` |
| regex PII | Presidio / a cloud DLP API | `gateway/redaction.py` |
| in-code policy rules | OPA / Cedar | `policy/engine.py` |
| HMAC agent tokens | SPIFFE / AgentCore Identity | `identity.py` |
| operator tokens in an env var | SSO (OIDC) | `main.py` (`operator`) |
| JSONL traces | OTLP (already supported) | `observability.py` |
| synthetic entity graph | Neo4j + a CDC pipeline | `tenants/fincrime/data.py` |
