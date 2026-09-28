# Interview Q&A - answered against this codebase

The questions an Agentic AI Engineer interview actually asks, with answers you can give
out loud and back with a file you built. Rehearse with the code open. Every answer is
deliberately 3-5 sentences - interview length, not essay length.

---

**Q1. Why the Claude Agent SDK instead of calling the Anthropic API directly?**

Because with the raw API, *I* own the agent loop - tool-call parsing, context compaction
when the window fills, error recovery, retries, loop detection, termination - and that
loop is where most production reliability bugs live. The SDK ships Claude Code's
battle-tested loop as a library, so my code is only the parts specific to my product:
tools, guardrails, review, orchestration, evals. In this repo the proof is
[`sdk_agent.py`](../backend/src/harness/sdk_agent.py): it contains **no loop at all** -
it registers tools and hooks, sets budgets, and calls `query()`.

**Q2. What is an "agent harness"?**

The runtime scaffolding that turns a stateless text-in/text-out model into a system that
pursues a goal over many steps in an environment. Concretely: the gather-context ->
act -> verify loop, context management, tool dispatch, error recovery as observations,
termination and budgets, and persistence. The model supplies judgment; the harness makes
that judgment usable repeatedly and safely. In the Claude Agent SDK, the harness ships in
the box - that is the product.

**Q3. So where is the harness in your project?**

Inherited, not written. The SDK runs the loop; what I wrote is the *contract around it*
([`harness/base.py`](../backend/src/harness/base.py)): a tool catalog exposed as an
in-process MCP server, PreToolUse policy hooks, hard budget ceilings, and an event
stream. My deterministic offline worker honours the identical contract, which is why CI
exercises the whole system for free.

**Q4. Claude Agent SDK vs LangGraph - which one and why?**

They live at different layers, so: both. LangGraph is *explicit* orchestration - you
draw the control flow; the SDK is *emergent* - the model drives inside a managed loop.
Per-repo migration is open-ended (find every occurrence, iterate on test failures), so
you cannot draw it as a graph - that is an SDK agent. The fleet level (which repo, what
order, gated how, aggregated how) is explicit and durable - that is my LangGraph
`StateGraph` ([`orchestrator/graph.py`](../backend/src/orchestrator/graph.py)), whose
nodes contain no loop; they invoke SDK workers. LangGraph is the org chart, the SDK is
the employee.

**Q5. How do you stop an agent from looping forever and burning money?**

Hard ceilings enforced outside the model, never as prompt instructions:
[`Budget`](../backend/src/harness/base.py) raises `BudgetExceeded` on step, tool-call, or
dollar limits, and the worker converts that into a clean escalation to a human. Squeeze
`ME_MAX_AGENT_STEPS` and you watch repos escalate instead of thrash. Prompts are
suggestions; exceptions are physics.

**Q6. What is reward hacking and how do you defend against it?**

An agent optimizing the *signal* instead of the *goal* - the classic being "make tests
pass" by weakening or deleting the tests. I defend in three independent layers: a
PreToolUse hook blocks writes to any test file
([`guardrails/hooks.py`](../backend/src/guardrails/hooks.py)); a reviewer agent
re-derives what changed from `git diff` and vetoes tampering
([`harness/reviewer.py`](../backend/src/harness/reviewer.py)); and the eval gate tracks a
`no_tamper_rate` that must hold 100% or CI fails. I have an executable demo: disable
layer 1 and the test suite proves layer 2 still catches it
([`tests/test_defense_in_depth.py`](../backend/tests/test_defense_in_depth.py)).

**Q7. Why route every tool call through one choke point?**

Because policy enforced by convention decays; policy enforced by construction cannot be
bypassed. [`ToolInvoker.call`](../backend/src/tools/registry.py) is the only path to any
tool in every execution mode: hook, execute, redact secrets, meter the budget, emit an
event, record a trace span. New tools inherit all of it automatically.

**Q8. What is MCP and why does it matter here?**

An open protocol for exposing tools, data, and prompts to any agent - it turns tools
from code entangled in one agent into a reusable, independently secured capability
catalog. Here my repo/test/rules/memory tools become an in-process SDK MCP server via
`create_sdk_mcp_server` (see `sdk_agent.py`); in a multi-employee platform the same
tools would be external MCP servers shared by every agent.

**Q9. How do you evaluate an agent?**

Trajectory-level, not just final-answer, and as a *gate*, not a report.
[`evaluation/run_eval.py`](../backend/evaluation/run_eval.py) replays golden migrations
and exits non-zero on regression, so prompt/tool changes cannot ship if success rate,
`no_tamper_rate`, or PR-open rate drop. The process metric matters most: an agent can
look perfect on outcomes while cheating on process.

**Q10. Human-in-the-loop - where and why there?**

At the irreversible boundary: nothing is pushed or opened as a PR until a person
approves that specific diff. Implementation: the repo's task awaits an
`asyncio.Future` that the approval endpoint resolves
([`orchestrator/engine.py`](../backend/src/orchestrator/engine.py)); deliberately
*outside* the compute semaphore, so a repo waiting on a human never blocks another
repo's migration.

**Q11. What would you change to run this for real at scale?**

The adapters, not the architecture: worker in an egress-allowlisted sandbox
(Firecracker/gVisor) with a short-lived GitHub App installation token; JSONL traces to
OTLP/Langfuse; file memory to pgvector; in-process jobs to a durable queue with a run
store; the batch gate to a LangGraph `interrupt()` checkpoint. Every one swaps behind an
interface that already exists - that is what "production-faithful demo" means.

**Q12. When would you NOT use the Claude Agent SDK?**

Deterministic, compliance-bound workflows where "the agent decided" is unacceptable -
draw those in LangGraph or plain code. Non-Claude or model-agnostic requirements - the
SDK is Claude-only. And pure retrieval/chat, which needs no environment loop at all.
Naming the tool's limits is what makes the rest of the answers credible.

---

## If you get one whiteboard minute

Draw this and talk through it left to right:

```
human console ── approve/reject ──┐
                                  v
LangGraph / control plane ─► fan-out ─► [ per repo: SDK worker loop
   (explicit, durable)                     tools via MCP ─ hooks ─ budget ]
                                              │
                                    reviewer (tamper gate)
                                              │
                                    real git branch ─► real PR
        evals gate CI ◄── traces/events from every tool call
```

One sentence to close: "The SDK owns the loop; I own everything that makes the loop safe
to point at production."
