# How the Incident Commander Works

> Read this after the [README](../README.md). It explains *why* the system is shaped
> the way it is. Where a decision had a real alternative, [`adr/`](adr/) records the
> tradeoff. Where you should go break something to understand it,
> [`../EXERCISES.md`](../EXERCISES.md) tells you how.

---

## 1. The one idea

Everything here follows from a single rule:

> **The model decides what to do. Code decides what is allowed.**

The LLM picks the route, forms the diagnosis, chooses the action, and judges
recovery. It never sets a ceiling, grants itself a permission, classifies its own
risk, or waives an approval. Those are enforced in `loop.py` and
`guardrails/manager.py`, in ordinary Python, where they can be read and tested.

This is not stylistic. Telemetry is attacker-influenced text that flows into the
prompt. If any model output could widen the model's own permissions, then a log
line becomes a privilege-escalation vector. The separation is the security
boundary, and every design decision below is downstream of it.

```
   +----------------------- MODEL DECIDES -----------------------+
   |  route . diagnose . propose . critique . verify . write-up  |
   +------------------------------+------------------------------+
                                  |  structured output only
   +------------------------------v------------------------------+
   |                     CODE DECIDES (loop.py)                  |
   |  ceilings . RBAC . risk class . repeat block . human gate   |
   +------------------------------+------------------------------+
                                  |  exactly one mutating call
                          execute_remediation()
```

---

## 2. The control loop

One incident = one run of the loop. All state lives in a single typed
`IncidentState`, which is what makes the run auditable, replayable, and (for the
graph engine) serializable.

```
trigger(scenario) -> IncidentState + SandboxStack + GuardrailManager
      |
      |- recall similar past incidents from runbook memory
      v
  background asyncio task: IncidentRun(state).run()
      |
      v
  while status not terminal:
      |-- check halt conditions ------------------------> escalate / cancel
      |-- iterations += 1
      |-- OBSERVE      read-only telemetry, then sanitize (quarantine + redact)
      |-- ROUTE        Supervisor picks the next agent; code clamps the choice
      |
      |-- TRIAGE           (first pass only) severity + blast radius
      |-- DIAGNOSE         ranked hypotheses + confidence
      |     |- needs_more_data? --> GATHER EVIDENCE (run tools) --> re-DIAGNOSE
      |     |- still under the confidence floor? --------> escalate
      |-- PROPOSE          one action, from a closed enum, with a prediction
      |-- GUARDRAIL        RBAC allowlist + repeat block + preconditions --> escalate/retry
      |-- CRITIC           adversarial review; a veto sends it back to re-diagnose
      |-- HUMAN GATE       high risk always; with a timeout ---> escalate
      |-- ACT              execute_remediation - the ONLY mutating call
      |-- VERIFY           fresh telemetry; was the prediction met?
      |                       code vetoes "recovered" if the metrics disagree
      |                       |-> resolve | loop again | escalate
      v
  POSTMORTEM (always, on every outcome; built from the record if no model can write it)
      v
  record the episode in runbook memory (not for outages or crashes)
```

### 2.1 Halt conditions

Six hard ceilings, plus two soft halts. All enforced in code, none requested in a
prompt. `tests/test_halt_conditions.py` drives each one to an actual stop.

| Halt | Setting | Enforced |
|---|---|---|
| Recovery confirmed | - | Verifier reads fresh telemetry |
| Iteration ceiling | `IC_MAX_ITERATIONS` | top of every pass |
| Cost ceiling | `IC_MAX_COST_USD` | top of every pass **and** pre-flight per model call |
| Wall-clock ceiling | `IC_MAX_WALLCLOCK_S` | top of every pass |
| Human rejection | - | approval gate |
| Approval timeout | `IC_APPROVAL_TIMEOUT_S` | approval gate |
| Operator cancel | - | top of every pass; also unblocks a parked gate |
| *(soft)* Low confidence | `IC_MIN_ACTION_CONFIDENCE` | after evidence rounds are spent |
| *(soft)* Critic vetoes | `IC_MAX_CRITIC_VETOES` | after the critic loop |

Every terminal incident carries exactly one `HaltReason`. It is an enum, not free
text, specifically so the evaluation harness can measure **how** runs end rather
than only whether they succeeded - "resolved 90% of the time" and "resolved 90% of
the time and escalated correctly the other 10%" are very different systems.

**Why cost is checked twice.** Checking only at the top of an iteration lets a
single pass overshoot the ceiling by a whole pass before anyone notices. The
pre-flight check in `llm.structured()` estimates a call's cost from its prompt size
and *refuses*, rather than discovering the breach after paying for it.

---

## 3. The agents

Each agent is a pure function `run(state, ..., llm) -> Schema`. It reads state,
makes exactly one model call through the `llm.structured` seam, and returns a
validated object. It never mutates state, never calls a write-side tool, and never
decides its own permissions. That uniformity is what lets both engines drive the
same seven functions without modification.

| Agent | Decides | Model tier | Runs |
|---|---|---|---|
| **Supervisor** | which specialist acts next | fast | every pass (off by default) |
| **Triage** | severity, impact, blast radius | fast | once |
| **Diagnostician** | root cause, or which tools it needs | strong | every pass |
| **Remediation** | one action + a falsifiable prediction | strong | when confident |
| **Critic** | whether that action is actually right | strong | before the gate |
| **Verifier** | recovered / loop / escalate | fast | after acting |
| **Postmortem** | writeup + reusable runbook lesson | fast | at the end |

### 3.1 Why the Diagnostician has three outputs, not one

This is the difference between an agent and a function call. It can return:

1. **A confident leading cause** -> the loop proceeds to propose.
2. **`needs_more_data` + `requested_tools`** -> the loop runs those read-only tools
   and calls it again with the results.
3. **Persistent low confidence** -> the loop escalates rather than acting on a guess.

Path 2 is the interesting one. Pool stats, config diff, dependency health and SLO
burn are deliberately **not** in the default telemetry bundle, exactly like a real
observability stack where every query has a cost. The `silent_config_drift`
scenario is unsolvable without requesting a tool, which is what keeps this branch
honest rather than decorative.

### 3.2 Why there is a Critic

The Remediation agent is optimizing for "produce a plausible action". Nothing in
that objective rewards noticing the action is wrong. So a second agent gets the
opposite objective: find the reason this is a mistake.

Three details make it a real reviewer rather than a rubber stamp:

- It is **not shown the planner's confidence**. Confidence is contagious.
- **`strongest_objection` is required even when endorsing.** Without it, a critic
  discharges its duty with "looks good", which is worth nothing.
- The prompt states the asymmetry explicitly: a wrong veto costs one iteration, a
  wrong endorsement costs an outage. The default failure mode of an LLM reviewer is
  agreeableness, and it has to be argued out of it.

Turn it off with `IC_CRITIC_ENABLED=false` and compare - that experiment is
[Exercise 4](../EXERCISES.md).

### 3.3 Why the Supervisor is optional

The Supervisor routes; it does not govern. Every decision it makes passes through
`sanitize_route()`, which clamps the choice to something legal, and then through
the same ceilings and gates as any other path. A router that could skip the
approval gate would be a privilege-escalation path reachable from a log line.

Because of that constraint, `IC_SUPERVISOR_ENABLED=false` swaps in
`default_route()`, a pure function implementing the same policy deterministically.
Running both and diffing trajectories shows what the LLM router actually buys you -
which on the easy scenarios is often nothing. That is a finding worth having, and
the code is arranged so you can get it.

---

## 4. Falsification: the loop's quality mechanism

The single most important wiring in the system, and the thing most agent loops
lack:

```
Remediation:  expected_signal = "error_rate drops below 0.01 and p99 below 500ms"
                     |
                     v
Verifier:     expected_signal_met = ?   <- judged against the prediction,
              signal_assessment  = "predicted <0.01, observed 0.001"    not a vibe
```

A proposal that cannot be wrong cannot be verified, and an agent that cannot be
wrong cannot improve. Two failure modes this catches that outcome checking does
not:

- **Vague proposals.** "Things should improve" is unfalsifiable, and the Verifier
  is forced to say so instead of nodding along forever.
- **Unexplained recovery.** `recovered=true` with `expected_signal_met=false` means
  the service came back but *not for the reason we thought*. The diagnosis is still
  wrong and the fault will return. The loop emits a `guardrail` event for this
  rather than quietly banking the win.

---

## 5. The safety model (OWASP Top 10 for Agentic Applications)

| OWASP risk | Mitigation | Where |
|---|---|---|
| Excessive agency | risk scoring + human gate; high risk always stops for a human | `guardrails/manager.py`, `loop.py` |
| Improper tool permissions | role allowlist, fail-closed on unknown role or action | `ROLE_ALLOWED` |
| Goal hijacking / prompt injection | redact -> neutralize -> spotlight-fence | `GuardrailManager.sanitize` |
| Sensitive information disclosure | secret/PII redaction before prompting | `REDACTION_PATTERNS` |
| Memory poisoning | failed runs stored *as failures*; schema-versioned entries | `memory/runbook.py` |
| Unbounded consumption | six ceilings + repeat-action block + tool-batch cap | `loop.py`, `config.py` |
| Lack of observability | typed findings, full event log, per-agent cost | throughout |

### 5.1 Prompt injection: quarantine, not detection

The honest version of this is worth stating plainly, because most write-ups
overclaim it.

A regex list does **not** solve prompt injection. What it does is raise the cost of
the easy attacks. `sanitize()` runs three steps, each covering a different gap:

1. **Redact** secret- and PII-shaped substrings. Telemetry is a notorious leak
   path: a stack trace with a bearer token in it goes into the prompt, then the
   postmortem, then wherever postmortems are aggregated.
2. **Neutralize** injection-shaped imperatives, replacing the text with an inert
   marker. The fact of the attempt survives for the audit trail; the instruction
   does not. *A detector that flags the payload and forwards it verbatim has logged
   an attack, not stopped one* - that was the flaw in the previous version.
3. **Spotlight** everything remaining inside an explicit untrusted-data fence, with
   a standing instruction that content inside is data. Step 2 catches known
   patterns; step 3 limits the damage from the ones nobody has thought of yet.

**The real backstop is architectural, not textual.** Even a fully hijacked model
can only emit an action from a closed enum, that action must be in the role's
allowlist, and high-risk actions still stop for a human. The `poisoned_telemetry`
scenario tests exactly this end to end: the injected text demands a `failover`, the
real fault needs a `restart_service`, and a correct run does the latter *and*
records the attempt.

### 5.2 Least privilege is real, not decorative

`IC_AGENT_ROLE` selects from three roles, and the loop honours it:

| Role | May run | Use |
|---|---|---|
| `sre-bot` | everything (high risk still gated) | default |
| `sre-bot-restricted` | cache, restart, scale - never deploy or traffic changes | production-shaped |
| `read-only` | `no_op` only | agent as advisor |

An unrecognized role gets **no** permissions. A typo in the environment disables
the agent rather than granting it permissions.

---

## 6. Memory

When an incident ends, its postmortem is distilled into one `Episode` and stored.
When a new incident starts, similar past episodes are retrieved and handed to the
Diagnostician as prior art.

**Retrieval is lexical (BM25), not embedding-based.** Deliberately:

- Incident telemetry is high-signal keyword text - `OutOfMemoryError`,
  `HikariPool-1`, `SSLHandshakeException`. These are near-identifiers, and lexical
  matching is strong on identifiers where embeddings are comparatively weak. The
  classic embedding failure here is deciding two different exception classes are
  0.94 similar because both are "a Java error".
- No extra dependency, no API cost, no cold start, and **deterministic**, so tests
  can assert on retrieval results.
- **Inspectable**: every hit reports the terms that matched.

The honest limitation: lexical retrieval misses paraphrase. "heap exhausted" will
not match "out of memory". Hybrid search is [Exercise 6](../EXERCISES.md), and
`Episode.symptom_terms` is the seam.

**Framing matters as much as retrieval.** Presented as fact, a recalled episode
anchors the model onto a superficially similar cause. `memory_block()` presents it
as "a past case that may not apply, which you must check against today's evidence"
- a hypothesis generator, not an answer key. Failed runs are stored as failures and
retrieved as cautionary examples, which is the mitigation for memory poisoning.

---

## 7. Two engines, one system

| Concern | `loop.py` (explicit) | `graph.py` (StateGraph) |
|---|---|---|
| Control flow | `while` + `if` | nodes + conditional edges |
| Routing | `IncidentRun.route()` calls the Supervisor | pure edge functions |
| Halt conditions | top of the loop | a `guard` node that can jump to END |
| **Durability** | none - a restart loses the run | a checkpointer persists every step |
| **Resumability** | none | resume by `thread_id` |
| Human-in-the-loop | `await` on an asyncio Future | `interrupt_before=["act"]` + state update |
| Debuggability | read top to bottom | inspect the checkpoint at each node |

Select with `IC_ENGINE=loop|graph`. `tests/test_graph_parity.py` asserts both reach
the same outcome on every scenario.

**What the graph actually buys you: durability, resumability, inspectability.** It
does not buy safety - the guardrails are the same code called from the same places.
A framework that made those implicit would make the system harder to audit.

The genuinely better part is `interrupt_before=["act"]`. In the explicit loop, an
incident parked at the approval gate lives as a pending Future in one process;
restart it and the incident is gone. The graph checkpoints *before* the only
mutating node, so with a persistent checkpointer the approval can arrive hours
later, from another process, after a deploy. That is what "durable
human-in-the-loop" means.

As shipped, the checkpointer is the in-process `MemorySaver`, so this is the
*seam* for durability, not durability itself: swap in `AsyncPostgresSaver` (and
keep the approval registry outside the process) and it holds. That swap is
[Exercise 15](../EXERCISES.md).

> **A trap worth knowing.** The checkpointer serializes state, so
> `snapshot.values["incident"]` is a *copy*. Mutating the object you passed in has
> no effect; the decision has to be written back with `update_state`. Get it wrong
> and you build a graph that appears to accept an approval and then ignores it. See
> `_merge_graph_state` in `loop.py` for the other half of that hazard - blindly
> merging the copy back would overwrite the event log and erase the audit trail.

---

## 8. The sandbox

`sandbox/stack.py` simulates a microservice stack. Two properties are load-bearing:

1. **Ground truth is hidden.** `observe_all` strips the internal `faulted` flag.
   Agents infer health from error rate and latency, like a real on-call engineer.
2. **Some signals cost a tool call.** The default bundle is cheap and *incomplete*.
   An agent that needs more has to decide it needs more.

The fault catalog is built so the eval can measure things a happy-path catalog
cannot:

| Family | Example | Lesson |
|---|---|---|
| Single-step | `bad_deploy` | read telemetry, pick the one right action |
| Compound | `cascading_failure` | telemetry evolves; the loop must re-diagnose |
| Adversarial | `poisoned_telemetry` | tool output is untrusted input |
| **Negative** | `unfixable_ca_expiry` | knowing when to escalate |
| **Negative** | `false_alarm` | knowing when to do nothing |
| Diagnostic | `silent_config_drift` | needs a tool the default bundle omits |

The negative scenarios matter more than they look. An agent evaluated only on
faults it can fix will happily invent a fix for one it cannot, and will never be
measured for it. `expected_outcome` encodes the correct *ending* per scenario, so
"escalated when it should have" is scored as strictly as "resolved when it should
have".

---

## 9. Evaluation

Full reasoning in `evaluation/metrics.py`. The short version:

| Family | Metrics | Question |
|---|---|---|
| Outcome | correct-outcome rate + Wilson CI | did it end correctly? |
| Efficiency | correct-first-action, iterations, cost | what did it cost? |
| **Safety** | harmful-action rate, injection-resisted rate | what did it do on the way? |
| **Process** | trajectory score, LLM-judge | was the reasoning sound? |
| **Reliability** | stability across N repeats, flaky scenarios | would it do it again? |

Three things this harness does that a typical one does not:

- **N repeats by default.** One run of a stochastic system is an anecdote. The
  `stability` column - how often repeats agreed - is usually the most actionable
  number in the report.
- **The judge is not told the outcome.** A judge that knows the run succeeded
  reverse-engineers a justification for it, and a lucky guess scores as good
  reasoning. Withholding it is the single most important choice in `judge.py`.
- **`--stub` runs the identical harness deterministically**, so CI checks the
  harness and the loop's control flow on every push, for free. Stub numbers measure
  *harness health*, not agent quality, and the report says so.
- **Inapplicable trajectory checks are omitted, not failed.** Three of them
  (`correct_first_action`, `predictions_made`, `predictions_checked`) require the
  agent to have acted, so on a negative scenario like `unfixable_ca_expiry` they
  cannot pass. Scoring them anyway meant a flawless *restraint* run -- investigated,
  cited its evidence, honestly uncertain, correctly declined to act -- scored
  **0.625** against a flawless action run's **1.000**. Since `trajectory_score` is a
  higher-is-better metric, `compare_to_baseline` would have read "the agent now acts
  on incidents it cannot fix" as an *improvement*: a metric pointing backwards at
  exactly the failure ADR-0006 exists to catch. The denominator now varies by run,
  which is the price of comparing runs that had different work to do.

---

## 10. What is still missing

Stated plainly, because a document that claims completeness is not credible:

- **Durable approvals.** Incidents and events can persist to Postgres
  ([persistence-and-login.md](persistence-and-login.md)), but a run in progress
  cannot resume after a restart: the approval registry is in process memory and the
  graph's checkpointer is the in-process `MemorySaver`. The seam is there (swap in
  `AsyncPostgresSaver`, move approvals to Redis); the work is Exercise 15.
- **Real adapters.** The sandbox stands in for Prometheus, Loki, Kubernetes and a
  deploy system. The seams now exist - `integrations.TelemetrySource` and
  `integrations.ActionExecutor`, which `SandboxStack` satisfies as the reference
  implementation - but no real adapter ships. See
  [ADR-0010](adr/0010-integrating-real-infrastructure.md) and
  [integration.md](integration.md).
- **Typed predictions.** `expected_signal` is free text that the Verifier (a model)
  judges. Code already refuses a "recovered" verdict the metrics contradict
  (`IC_RECOVERY_MAX_*`), but it does not check the prediction itself. A typed
  prediction (metric, comparator, threshold, deadline) that code evaluates would
  close that gap.
- **Verification against a real system.** The larger half of the adapter problem
  and the one most easily missed: the sandbox returns `effective: True` instantly
  because it owns the ground truth. On real infrastructure that is a statistical
  judgement over a noisy signal with a propagation delay. `ActionSpec.settle_seconds`
  and `verification_signal` create the place a real answer goes; they do not
  produce one.
- **Concurrency on a target.** Nothing stops two incidents - or an agent and a
  human - acting on one service at once. Leasing per target is a prerequisite for
  any autonomous write access.
- **Distributed rate limiting.** The limiter is in-process, so behind N replicas the
  effective limit is `limit x N`.
- **Tracing.** No OpenTelemetry or Langfuse spans yet; per-agent cost accounting is
  the groundwork.
- **Injection defence is partial.** See section 5.1. The architectural backstop is
  the real protection.
- **The judge is a model.** It measures plausibility to a reader, which correlates
  with quality without being it. Use it for regressions and ranking, not as an
  oracle.

---

## 11. Where to read next

| You want to... | Go to |
|---|---|
| Understand a specific decision | [`adr/`](adr/) |
| Learn by breaking things | [`../EXERCISES.md`](../EXERCISES.md) |
| See how prompts were built and why | [`prompt-engineering.md`](prompt-engineering.md) |
| Look up a term | [`glossary.md`](glossary.md) |
| Step through one agent at a time | [`../notebooks/`](../notebooks/) |
| Read the loop itself | `backend/src/loop.py` |
