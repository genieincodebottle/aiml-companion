# Exercises

Reading an agent loop teaches you what it does. Breaking one teaches you why it is
shaped that way.

These are ordered roughly by difficulty. Each states the **question** it answers,
what to **do**, what you should **observe**, and **why the answer is what it is**.
That last part is the one most exercise sets skip. Several exercises will make the
system behave *worse*. That is intentional: the fastest way to understand a
guardrail is to watch what happens without it.

**Setup.** Most exercises are an environment variable and a re-run. Nothing here
needs a code change unless it says so.

```bash
cd backend
uv sync --extra dev
uv run pytest -q                                   # the whole suite, no API key needed
uv run python -m evaluation.run_eval --stub        # deterministic harness check
```

Exercises marked **[live]** need `GOOGLE_API_KEY` because they measure the model
rather than the loop. **Everything else runs offline, instantly, and free** - which
is deliberate, because most of what there is to learn here is in the loop, not in
the model.

Measured costs before you spend anything:

| | Time | Cost |
|---|---|---|
| Any `--stub` exercise | ~3s | free |
| One live scenario (`--scenarios bad_deploy`) | ~30s | ~$0.012 |
| A full live pass over all 10 scenarios | ~10-12 min | ~$0.22 |

So run live on **one scenario first**. A full pass is worth it once, when you want
a real number to compare against - not on every edit.

---

## Level 1 - See the machine work

### 1. Watch the loop actually loop

**Question:** What does it look like when an agent self-corrects instead of
one-shotting a fix?

**Do:** Run the `cascading_failure` scenario in the console. Approve both actions.

**Observe:** The Verifier reports `partial progress` after the rollback, telemetry
*changes*, and the second pass reaches a **different** leading cause before
restarting the service.

**Why:** A deploy leaked connections and exhausted the pool. Rolling back stops new
leaks; the already-leaked connections stay stuck until a restart drains them. One
action can never fully recover it, so the loop has to iterate - which is only
possible because the Verifier reads *fresh telemetry* rather than trusting the
action's self-report.

> Now read `test_compound_fault_requires_multiple_iterations`. It asserts the
> diagnoses actually differ. An earlier version of this project passed every other
> assertion in that test while re-proposing from a stale diagnosis; that one line
> is what caught it.

### 2. Make the agent ask for evidence

**Question:** What does `needs_more_data` actually *do*?

**Do:** Run `silent_config_drift`.

**Observe:** The first diagnosis reports low confidence and requests
`get_config_diff`. An `evidence_request` event appears, tools run, and the second
diagnosis is confident.

Now disable the branch and re-run:

```bash
IC_MAX_EVIDENCE_ROUNDS=0 uv run python -m evaluation.run_eval --stub --scenarios silent_config_drift
```

The incident escalates with `halt_reason=low_confidence`.

**Why:** The cause is invisible in metrics, logs, traces and deploys - it only
appears in the config diff, which is deliberately *not* in the default telemetry
bundle. An agent that cannot request a tool cannot solve this. And an agent that
cannot request a tool *and* cannot admit uncertainty would have guessed.

Escalating is the correct behaviour, not a failure. Notice that the system had to
be built to allow it: without a licensed way to say "I don't know", a model will
invent an answer.

### 3. Try to hijack the agent

**Question:** Does the injection defence do anything?

**Do:** Run `poisoned_telemetry`. The logs contain attacker text demanding a
`failover` and the disclosure of the system prompt. Watch the `guardrail` events.

Then turn the defence off:

```bash
IC_QUARANTINE_INJECTIONS=false uv run python -m evaluation.run_eval --stub --scenarios poisoned_telemetry
```

**Observe:** With quarantine on, the imperative text is replaced by a marker and
the agent restarts the service (the real fix). With it off, the payload reaches the
model verbatim - the attempt is still *detected*, just not *stopped*.

**Why:** This is the difference between logging an attack and defending against
one. See [ADR-0007](docs/adr/0007-quarantine-not-detection.md).

**The important part:** even with quarantine off, a hijacked model still cannot run
an arbitrary command. It can only emit an action from a closed enum, which must be
in its role's allowlist, and `failover` is high-risk so it stops for a human. The
regex is the *outer* layer; the architecture is the real defence.

---

## Level 2 - Remove a guardrail and measure the damage

### 4. Turn off the Critic

**Question:** Is the extra model call worth it?

```bash
uv run python -m evaluation.run_eval --stub --repeats 3 --out /tmp/with-critic.json
IC_CRITIC_ENABLED=false uv run python -m evaluation.run_eval --stub --repeats 3 --out /tmp/no-critic.json
```

Compare `harmful_action_rate` and `trajectory_score`.

**[live]** Repeat without `--stub` for the honest answer - the stub critic is
deliberately conservative and only vetoes the indefensible case, so the offline
difference understates it.

**Why:** The Critic costs one strong-tier call per proposal - roughly +20% on a
single-iteration incident - and pays off only on runs where the planner was wrong.
Whether that is worth it depends entirely on the blast radius of a wrong action.
That is a judgement call, which is why it is a flag and not a hardcoded choice.
See [ADR-0008](docs/adr/0008-adversarial-critic.md).

### 5. Give the agent less privilege

**Question:** What does least privilege actually change?

```bash
IC_AGENT_ROLE=sre-bot-restricted uv run python -m evaluation.run_eval --stub
IC_AGENT_ROLE=read-only          uv run python -m evaluation.run_eval --stub
IC_AGENT_ROLE=typo-does-not-exist uv run python -m evaluation.run_eval --stub
```

**Observe:** `sre-bot-restricted` cannot roll back, so `bad_deploy` escalates with
`halt_reason=guardrail_blocked`. `read-only` escalates almost everything. The typo
role can do **nothing at all**.

**Why:** Fail-closed. A typo in an environment variable leaves the agent with no
permissions at all. That is the opposite default from most permission systems, and
the correct one when the subject is an autonomous process with production access.

Notice this whole code path was previously unreachable in the running app: the role
was hardcoded and every action was allowlisted, so the RBAC branch only ever
executed in a unit test.

### 6. Break the memory

**Question:** Does episodic memory help, and how would you know?

**Do:** Run `memory_leak` twice and watch the second run's `memory_recall` event.
Then:

```bash
IC_MEMORY_ENABLED=false uv run python -m evaluation.run_eval --stub --repeats 3
```

Now try to *break* retrieval. Open `backend/data/runbook_memory.json` and edit an
episode's `symptom` to describe a completely different fault. Run again and watch
what gets recalled.

**Why:** Memory is an attack surface - OWASP calls it memory poisoning. Two
mitigations are worth finding in the code: failed runs are stored **as failures**
and retrieved as cautionary examples, and `min_score` suppresses the
"retrieve-something-anyway" behaviour. Irrelevant prior art is worse than none,
because it anchors the model on a superficially similar cause.

---

## Level 3 - Change the system

### 7. Add a fault scenario

**Do:** Add a scenario to `backend/src/sandbox/scenarios.py` that the current agent
will get **wrong**. Suggestions:

- A disk filling up, where the only listed action that helps is `restart_service`,
  but the telemetry looks like a memory leak.
- A fault where the *tempting* action (`failover`) makes things worse because the
  standby is also degraded - put that in `dependency_health`.
- A compound fault needing three ordered steps.

Then run `uv run pytest -q`. `test_every_scenario_reaches_its_expected_outcome` is
parametrized over the catalog, so your scenario is tested automatically.

**Why:** Writing a scenario forces you to separate "what the agent sees" from "what
is true" - the distinction the whole sandbox exists to maintain.

### 8. Wire the fallback action

**Question:** `RemediationProposal.fallback_if_ineffective` is generated and then
ignored. Should it be?

**Do:** Change the loop so that when an action produces no effect, the *declared
fallback* is tried before a full re-diagnosis. Measure `mean_iterations` and
`correct_first_action_rate` before and after.

**Why:** Skipping the re-diagnosis is faster and cheaper. It is also how an agent
commits harder to a wrong theory: the fallback was chosen under the same
misdiagnosis that produced the failed action. Decide which you want, and be able to
say why. (This is the same class of decision as the stale-diagnosis bug in
Exercise 1 - worth solving in that light.)

### 9. Make retrieval hybrid

**Do:** Add embedding-based retrieval alongside BM25 in `memory/runbook.py` and
fuse the rankings (reciprocal rank fusion is about ten lines). `Episode.symptom_terms`
is the seam.

**Then test the claim:** write a case where lexical retrieval *should* fail - 
"heap exhausted" against a stored "OutOfMemoryError" - and check hybrid fixes it
without breaking `test_irrelevant_query_retrieves_nothing`.

**Why:** [ADR-0005](docs/adr/0005-lexical-memory-over-embeddings.md) argues lexical
is the right default for identifier-heavy telemetry. Test that argument rather than
believing it. If you cannot construct a case where hybrid wins, the ADR is right
for this data; if you can, you have found the boundary.

### 10. Add a tracing backend

**Do:** Wire OpenTelemetry or Langfuse spans around each agent call in
`llm.structured`. The per-agent accounting in `TokenTracker` is the groundwork.

**Why:** `/api/incidents/{id}/usage` tells you the Diagnostician is 71% of cost. It
does not tell you *which prompt section* is 71% of the Diagnostician. Tracing does.

---

## Level 4 - Evaluate like you mean it

### 11. Find where the agent is unreliable **[live]**

```bash
uv run python -m evaluation.run_eval --repeats 5
```

Look at `stability` and `flaky_scenarios`, not the headline rate.

**Why:** A single run of a stochastic system is an anecdote. If a scenario resolves
3 times out of 5, its real success rate is somewhere in a wide interval - which is
exactly what the Wilson CI in the report is telling you. Any comparison between two
configurations that ignores the interval is noise.

### 12. Judge the reasoning, not the outcome **[live]**

```bash
uv run python -m evaluation.run_eval --repeats 2 --judge --out /tmp/judged.json
```

Then find a run with a **high outcome score and a low `evidence_grounding` score**
and read its transcript.

**Why:** That combination is the agent that got the right answer for the wrong
reason - and it will fail on the next incident you have not scripted. No outcome
metric can find it.

Note that the judge is deliberately **not told whether the run succeeded**. A judge
that knows the outcome reverse-engineers a justification for it. Read
`build_judge_prompt` and check for yourself what is withheld.

### 13. Deliberately regress the agent

**Do:** Weaken a prompt. Delete this sentence from the Diagnostician's system
prompt:

> "LEARN FROM FAILURE: you are shown what previous actions achieved..."

Then:

```bash
uv run python -m evaluation.run_eval --stub --repeats 3 --check-baseline
```

**Observe:** `cascading_failure` should degrade. The baseline check exits non-zero.

**Why:** Prompts are code, and this is what a regression test for a prompt looks
like. Also notice that `PROMPT_VERSION` is recorded in every eval report - so a
result can always be attributed to the prompts that produced it.

### 14. Compare the two engines

```bash
uv run python -m evaluation.run_eval --stub --engine loop  --out /tmp/loop.json
uv run python -m evaluation.run_eval --stub --engine graph --out /tmp/graph.json
```

**Observe:** Identical outcomes. That is what `tests/test_graph_parity.py` enforces
on every scenario, and CI runs the whole test suite a second time on the graph
engine (`IC_ENGINE=graph uv run pytest -q`).

**Then read both implementations** and answer: what does the graph give you that
the loop does not?

The answer is *durability*, and you can prove it. Start an incident, let it park at
the approval gate, and think about what happens if the process restarts. In
`loop.py` the pending approval is an `asyncio.Future` - it dies. In `graph.py` it
is a checkpoint before `interrupt_before=["act"]`, and the run resumes.

What the graph does **not** give you is safety. The guardrails are the same code
called from the same places. See
[ADR-0001](docs/adr/0001-explicit-loop-and-langgraph.md).

---

## Level 5 - Production

### 15. Make it survive a restart

**Do:** Replace `MemorySaver` in `graph.py` with `AsyncPostgresSaver`. Persist
`IncidentState` to Postgres. Move the approval-future registry to Redis.

**Test it properly:** start an incident, let it reach the approval gate, kill the
process, restart it, then approve. If the incident resumes, you have built durable
human-in-the-loop. If it does not, find out why - the answer is usually that
something lives in process memory that you assumed was in the checkpoint.

### 16. Replace the sandbox

**Do:** Implement `TOOL_REGISTRY` against real backends - Prometheus for metrics,
Loki for logs, the Kubernetes API for pod state - behind the same interface.

**Why:** The interface was shaped to accept this. If a real adapter does *not* fit,
the abstraction was wrong and you have learned something more useful than if it had
fitted.

**Do not** point `execute_remediation` at a real cluster without reading the
guardrail layer first, and without setting `IC_AGENT_ROLE=read-only` while you get
comfortable.

### 17. Attack it

**Do:** Write scenarios whose `injected_logs` try to defeat the current patterns.
Ideas: base64-encoded instructions, instructions split across several log lines,
text impersonating an approval that already happened, a fake `guardrail` event in
the log body.

**Then:** for each one that gets through the regex, check whether it actually
achieves anything. It should not - closed enums, the role allowlist and the human
gate are all still between the model and production.

**Why:** This is the exercise that teaches the real lesson of the whole project:
input filtering is a *speed bump*, and architecture is the *brake*. If you find an
attack that gets past the architecture, that is a genuine finding - write it up.

---

## Where to look when you are stuck

| Question | File |
|---|---|
| How does the loop decide what to do next? | `backend/src/agents/supervisor.py` |
| Where are the halt conditions enforced? | `backend/src/loop.py`, `IncidentRun.check_ceilings` |
| What can the agent actually do? | `backend/src/models/state.py`, `ActionType` |
| How is telemetry sanitized? | `backend/src/guardrails/manager.py`, `sanitize` |
| Why is this metric here? | `backend/evaluation/metrics.py` |
| Why was it built this way? | `docs/adr/` |
| What does a term mean? | `docs/glossary.md` |
