# Glossary

Terms as they are used *in this codebase*. Where a word is used loosely in the
wider industry, the entry says what it means here specifically.

---

## Agent concepts

**Agent loop** - A bounded control loop that observes, reasons, acts, verifies, and
decides whether to continue. The distinguishing feature is not the LLM; it is the
**halt conditions**. A loop without them is a runaway process with a language model
attached. → `backend/src/loop.py`

**Agent** - Here, a pure function `run(state, ..., llm) -> Schema` making exactly
one model call. It never mutates state, never calls a write-side tool, and never
decides its own permissions. That uniformity is what lets both engines drive the
same functions. → `backend/src/agents/`

**Multi-agent vs pipeline** - A pipeline runs stages in a fixed order. A multi-agent
system has something *choosing* which specialist acts next, and specialists that
can disagree. The Supervisor and the Critic are what make this the second thing.
A five-stage chain with five prompts is still a pipeline.

**Supervisor / router** - The agent that picks the next specialist. Here it is
advisory: `sanitize_route()` clamps its choice to something legal, and it can be
replaced entirely by a deterministic policy. → ADR-0002

**Critic** - An agent whose objective is to find the reason a proposal is wrong.
Distinct from self-critique, which asks the author to review their own work and
mostly produces agreement. → ADR-0008

**Trajectory** - The *path* a run took, as opposed to where it ended. Two runs can
both resolve an incident with completely different trajectories, and only one of
them will generalize. → `evaluation/metrics.py:score_trajectory`

**Halt condition** - A rule that stops the loop, enforced in code. This system has
six hard and two soft. Every terminal incident carries exactly one `HaltReason`.

**Human-in-the-loop (HITL)** - A point where the loop blocks for a human decision.
Load-bearing detail: it must have a **timeout**. A gate that waits forever converts
a closed laptop into a permanently stranded incident.

**Durable HITL** - The same gate, but checkpointed, so with a persistent
checkpointer the approval can arrive hours later from a different process after a
restart. The graph engine provides the seam (it ships with the in-process
`MemorySaver`); the explicit loop has no seam at all. → ADR-0001

---

## This system's vocabulary

**Incident** - One run of the loop, start to terminal state. Carries a full typed
`IncidentState` and an ordered event log.

**Iteration** - One pass of the loop: observe → route → (agents) → act → verify.
Bounded by `IC_MAX_ITERATIONS`.

**Evidence round** - An extra read-only tool fetch triggered when the Diagnostician
reports `needs_more_data`. Bounded by `IC_MAX_EVIDENCE_ROUNDS`, because an
unbounded "I need more data" loop is an unbounded loop.

**Expected signal** - The specific telemetry change a proposal predicts. Makes the
proposal *falsifiable*, which is the precondition for the Verifier being able to
judge it. → ADR-0004

**Falsification loop** - Remediation predicts → action runs → Verifier checks the
prediction against fresh telemetry. Catches the case where the service recovered
but *not for the reason we thought*, which outcome checking alone cannot see.

**Halt reason** - Enum, not free text, so the eval can count how runs end. A system
that resolves 90% and escalates 10% correctly is very different from one that
resolves 90% and crashes 10%.

**Negative scenario** - A scenario where success means **not** applying a fix:
`unfixable_ca_expiry` (escalate) and `false_alarm` (do nothing). Without these, an
agent cannot be penalized for confidently fixing a non-problem. → ADR-0006

**Harmful action** - An action applied to a healthy service. In production this is
how an agent turns a false alarm into an outage. Tracked as its own eval metric.

**Episode** - One remembered incident in runbook memory: symptom, root cause, what
was tried, what worked, and the outcome. Failed runs are stored as failures.

**Cautionary episode** - A recalled episode whose outcome was not `resolved`,
presented to the agent as an example of what did *not* work.

---

## Safety

**Guardrail** - A constraint enforced in code, not requested in a prompt. If
violating it would be a security incident, it is not a prompt. → ADR-0002

**Prompt injection** - Attacker-controlled text in tool output that attempts to
redirect the agent. Here it arrives through log lines. → `poisoned_telemetry`

**Quarantine** - Removing injection-shaped imperatives and leaving an audit marker,
as opposed to merely detecting them. A detector that forwards the payload has
logged an attack, not stopped one. → ADR-0007

**Spotlighting** - Wrapping untrusted content in an explicit fence with a standing
instruction that it is data, never instructions. Defence in depth, not a solution.

**RBAC / least privilege** - `IC_AGENT_ROLE` selects an allowlist of actions.
Fail-closed: an unrecognized role gets **no** permissions, so a typo disables the
agent rather than granting it permissions.

**Fail closed** - When uncertain, choose the restrictive option. Unknown action →
HIGH risk. Unknown role → no permissions. Model unavailable → escalate.

**Read/write separation** - Every observability tool is side-effect free; exactly
one function (`execute_remediation`) mutates anything, and only after the guardrail
layer and (where required) a human have cleared it.

**Closed enum** - An output constrained to a fixed set at the decoding layer, so
"hallucinate a new action" becomes a decoding error rather than a runtime surprise.
→ ADR-0003

**Memory poisoning** - Corrupting an agent's memory so it retrieves and trusts a
wrong belief. Mitigated by storing outcomes truthfully and versioning the schema.

**Circuit breaker** - After N consecutive model failures, fail fast for a cooldown
window instead of letting every incident independently retry a dead provider.

---

## LLM plumbing

**Structured output** - Binding a model call to a Pydantic schema so the agent
receives a validated object rather than text to parse. → `llm.structured`

**Model tiering** - Routing cheap classification agents to a cheaper model and
reasoning agents to a stronger one. `IC_MODEL_FAST` opts in; off by default so the
out-of-the-box behaviour is predictable. → `config.FAST_TIER_AGENTS`

**Cost pre-flight** - Estimating a call's cost *before* making it and refusing if
it would breach the remaining budget. Checking only after the fact means
discovering the breach having already paid for it.

**Token ledger** - Per-incident *and per-agent* usage accounting. "It cost $0.008"
is not actionable; "the Diagnostician is 71% of it" is. → `TokenTracker`

**Prompt version** - `PROMPT_VERSION`, bumped when a system prompt changes meaning
and recorded in every eval report, so a result can be attributed to the prompts
that produced it.

---

## Evaluation

**LLM-as-judge** - Using a model to score another model's reasoning against a
rubric. Here the judge is **not told the outcome**, because a judge that knows a run
succeeded reverse-engineers a justification for it. → `evaluation/judge.py`

**Correct-outcome rate** - The headline metric. Scored against each scenario's
`expected_outcome`, so correct escalation counts as success. Distinct from
resolution rate, which is reported separately for positive scenarios only.

**Correct-first-action** - Did the agent pick the right fix *first*? Separates "got
there" from "got there efficiently".

**Stability** - Fraction of repeats that agreed on the outcome. Usually the most
actionable number in the report, because a single run of a stochastic system is an
anecdote.

**Flaky scenario** - One whose repeats disagreed. Named explicitly rather than
averaged away.

**Wilson interval** - A confidence interval for a proportion. Reported because "5/6
resolved" is not 83% ± nothing, and a comparison that ignores the interval is
comparing noise.

**Baseline** - A committed report that later runs are diffed against, with a
tolerance. Without tolerance every run "regresses" on noise, the check gets ignored,
and then it catches nothing when something real breaks.

**Stub mode** - `--stub` runs the identical harness with a deterministic fake model.
Measures **harness health and control flow**, never agent quality. The report says
so, because conflating the two would be worse than not running it.

---

## SRE terms

**SEV1–SEV4** - Severity, worst to least. Rubric in the Triage prompt.

**Blast radius** - Which users and services an incident (or an action) affects.

**MTTR** - Mean time to resolution. Approximated here by mean iterations.

**Error budget / burn rate** - How fast an SLO's allowance is being consumed. A
burn rate above 14.4x exhausts a 30-day budget in under two days.
→ `get_slo_burn`

**Blameless postmortem** - A writeup describing systems and conditions, never
people. "The deploy pipeline had no canary stage", not "the engineer skipped
testing".

**Runbook** - A documented procedure for a known failure. Runbook *memory* here is
the agent building its own, one incident at a time.

**Cascading failure** - A fault that triggers further faults. The
`cascading_failure` scenario needs two ordered actions, which is what forces the
loop to genuinely iterate.

**Thundering herd / cache stampede** - Simultaneous cache misses overwhelming the
origin. → `cache_stampede`
