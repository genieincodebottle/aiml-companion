# Learning Guide

How to *study* this project, not just run it. Work through this guide and you will be
able to explain - and defend under questioning - what an agent harness is, why teams
build on the Claude Agent SDK instead of the raw API, where LangGraph fits, and how a
production team keeps an autonomous agent honest.

**Prerequisites:** comfortable Python, basic async, and having run the Quick Start once.
**Time:** ~3-4 hours for the reading path, more with the [exercises](EXERCISES.md).

---

## 1. The one idea this project exists to teach

A language model is a stateless function: text in, text out. An **agent** is a system
that pursues a goal over many steps by acting in an environment and reacting to results.
Everything that turns the first into the second - the loop, context management, tool
dispatch, error recovery, termination - is the **agent harness**.

The thesis, in one sentence:

> The harness is the hardest generic part of an agent, so inherit it (Claude Agent SDK)
> and spend your engineering budget on what is specific to you: tools, guardrails,
> review, orchestration, and evals.

This repo contains both sides of the argument on purpose:

| | [`incident-commander`](../../incident-commander/) | `migration-engineer` (this project) |
|---|---|---|
| The agent loop | **hand-written** (`src/loop.py`) - to teach *loop engineering* | **inherited** from the Claude Agent SDK - to teach *harness engineering* |
| What the code is mostly about | the loop itself: halt conditions, iteration, verification | everything *around* the loop: tools, hooks, budgets, review, fan-out, evals |
| Control flow | explicit: OBSERVE -> DIAGNOSE -> PROPOSE -> ACT -> VERIFY | emergent: the model decides the next action inside the SDK's loop |
| When each is right | bounded, auditable workflows | open-ended work in an environment (code, files, tests) |

Read them as a pair. Ask of every file: "who owns the loop here, and what did that
choice buy?"

## 2. The code-reading path (in this order)

Each step states what to look for. Do not skip ahead; the order is the argument.

1. **[`src/harness/base.py`](../backend/src/harness/base.py)** - the contract. One
   `WorkerContext`, one `Budget`, one `ToolSpec`. Notice there is NO loop here. Ask:
   why do budget ceilings raise an exception instead of returning a flag?
2. **[`src/harness/sdk_agent.py`](../backend/src/harness/sdk_agent.py)** - the live
   worker. Count the lines that implement a loop: zero. We build an in-process MCP
   server from our ToolSpecs, register a PreToolUse hook, set `max_turns`, and call
   `query()`. The SDK does the rest. This file IS the "why the SDK over the raw API"
   answer.
3. **[`src/harness/stub_agent.py`](../backend/src/harness/stub_agent.py)** - the
   offline stand-in. Key insight: it is scripted, but it drives the *identical* tools,
   hooks, budget, and events. That is why the offline demo is a faithful rehearsal of
   the live system, and why CI can gate on it for free. A third worker,
   [`src/harness/gemini_agent.py`](../backend/src/harness/gemini_agent.py), honours the
   same contract with a Gemini function-calling loop; the seam that picks between all
   three (SDK > Gemini > stub) is one function,
   [`src/harness/worker.py`](../backend/src/harness/worker.py)`::build_worker`.
4. **[`src/tools/registry.py`](../backend/src/tools/registry.py)** - the choke point.
   Every tool call, in every mode, flows through `ToolInvoker.call`: hook -> execute ->
   redact -> meter -> emit -> trace. Safety by construction, not convention.
5. **[`src/guardrails/hooks.py`](../backend/src/guardrails/hooks.py)** - the policy
   plane. Three checks: protected test files, path containment, secret redaction. Note
   that a blocked call does not crash the worker - the denial is returned as an
   observation the model must route around.
6. **[`src/harness/reviewer.py`](../backend/src/harness/reviewer.py)** - the second
   agent. It does not trust the migrator's self-report; it re-derives tamper status
   from `git diff`. Process verification beats outcome trust.
7. **[`src/orchestrator/engine.py`](../backend/src/orchestrator/engine.py)** - the live
   control plane: bounded-concurrency fan-out, SSE streaming, and the human PR gate
   implemented as an awaited `asyncio.Future`.
8. **[`src/orchestrator/graph.py`](../backend/src/orchestrator/graph.py)** - the same
   pipeline as a LangGraph `StateGraph`. Notice the nodes contain no agent loop; they
   call the same `migrate_repo_core`. LangGraph is the org chart, the SDK is the worker.
9. **[`src/vcs/`](../backend/src/vcs/)** - real git. `local.py` and `github.py`
   implement one interface, which is why the offline demo and your real GitHub repos are
   the same code path.
10. **[`evaluation/`](../backend/evaluation/)** - the promotion gate. Read
    `scorers.py` first: `no_tamper_rate` scores the *process*, not just the outcome.
    Then `run_eval.py`: it exits non-zero on regression, which is what makes it a CI/CD
    gate rather than a report.

## 3. An annotated trajectory

This is real output from `python cli.py stream datetime-fleet` (stub mode) - the CLI
passes `auto_approve=True` so the trajectory below runs straight through to `pr_opened`
with no `awaiting_approval` event; that event only fires when a job is NOT auto-approved,
which is how the React console runs it (see `orchestrator/engine.py`, the `if not
job.auto_approve` guard). Learn to read event streams like this - it is how you debug
agents in production.

```
[auth-service]  step             Starting 'Modernize deprecated datetime.utcnow()'
[auth-service]  tool_call        recall_memory -> 1 result(s)        <- cross-repo memory:
                                                                        an earlier repo already
                                                                        recorded the import gotcha
[auth-service]  tool_call        search_migration_rules -> 3 result(s)   <- RAG: pull the guidance
[auth-service]  tool_call        grep -> 3 match(es)                     <- locate targets
[auth-service]  step             Edit step 1/2: Replace datetime.utcnow()...
[auth-service]  tool_call        wrote auth.py (489 bytes)
[auth-service]  edit_applied     auth.py
[auth-service]  guardrail_block  write blocked: 'test_auth.py' is a
                                 protected test file                  <- LAYER 1: the hook refuses
                                                                        to let the sweep touch tests
[auth-service]  tests_run        FAIL                                 <- verification, not hope
[auth-service]  step             Tests still red; iterating.
                                 (FAIL: name 'timezone' is not defined)  <- the loop earns its keep:
                                                                        observe the error, adapt
[auth-service]  step             Edit step 2/2: Ensure `timezone` import
[auth-service]  tests_run        PASS
[auth-service]  review           approve=true tampered=false          <- LAYER 2: independent check
[auth-service]  pr_opened        local://auth-service/pull/1 (branch migration/...)
```

Three things to internalize: the agent *iterated on a real failure* (that is the whole
point of a loop); the guardrail fired *during* normal work, not in a contrived attack;
and - in the console, where jobs are NOT auto-approved - no change ships on the agent's
say-so alone; a human resolves the `awaiting_approval` gate first (LAYER 3).

## 4. Concepts checklist

You have understood the project when you can answer these without looking:

- Why does `sdk_agent.py` contain no `while` loop, and what five failure modes would a
  naive hand-rolled loop have that the SDK handles?
- Why is the stub "not a mock"? What exactly is identical between the three modes
  (stub / live-sdk / live-gemini), and what is the ONE thing that differs?
- Why is `ToolInvoker` a single choke point instead of letting each tool self-police?
- The reward hack: how could an agent make tests pass dishonestly, and what are the
  three independent layers that stop it here? (Run the lab in
  [EXERCISES.md](EXERCISES.md) to see layer 2 catch a real one.)
- Why does the LangGraph batch path deliberately NOT open pull requests?
- Why does the budget live outside the worker rather than being prompt instructions?
- What changes - and what does not - when you point this at real GitHub repos?

## 5. Where to go next

- Do the [exercises](EXERCISES.md) - especially #2 (write a rule end-to-end) and #3
  (the defense-in-depth lab).
- Rehearse [INTERVIEW_QA.md](INTERVIEW_QA.md) out loud, with the code open.
- Read Anthropic's "Building effective agents" and the Claude Agent SDK docs
  (https://code.claude.com/docs/en/agent-sdk/overview), then re-read `sdk_agent.py` -
  it will read differently the second time.
