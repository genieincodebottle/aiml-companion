# 0001. Ship the agent loop twice: explicit and as a StateGraph

**Status:** Accepted

## Context

This project has two jobs: be a working autonomous-SRE system, and teach loop
engineering. Those pull in opposite directions.

An explicit `while` loop makes every halt condition and the human `await` visible
on one screen - ideal for learning, useless in production, because a process
restart loses every in-flight incident. A `langgraph.StateGraph` gives durability,
resumability and step-level inspection - correct for production, but the control
flow becomes edges and a learner reading it sees a framework rather than a loop.

The previous version of this project claimed the node decomposition "maps 1:1 onto
a StateGraph" but never implemented it. `langgraph` was a dependency that was never
imported. That claim was unverifiable and, being untested, was drifting toward
false.

## Options

1. **Explicit loop only.** Best for reading. No durability. Leaves the README's
   framework claim unbacked.
2. **StateGraph only.** Production-shaped. Buries the loop's semantics in framework
   machinery; the teaching goal suffers.
3. **Both, driving the same agents, with a parity test.**

## Decision

Option 3. `loop.py` is the teaching reference and the default. `graph.py` is the
same system as a `StateGraph` with a checkpointer, selected with `IC_ENGINE=graph`.

Both drive the *same* seven agent functions and the *same* `GuardrailManager`.
`tests/test_graph_parity.py` asserts they reach the same outcome on every scenario.

## Consequences

**Good**

- The framework claim is now executable and tested rather than asserted.
- A learner can diff the two and see exactly what a graph framework adds - which
  turns out to be durability and inspectability, not safety.
- `interrupt_before=["act"]` demonstrates *durable* human-in-the-loop, which the
  explicit loop genuinely cannot do: a pending approval in `loop.py` is an
  `asyncio.Future` that dies with the process.
- The routing edges are pure functions of state, so they are unit-testable without
  running a graph.

**Bad - and this is the real cost**

- **Two code paths to maintain.** They can drift. The parity test is the mitigation,
  and it is load-bearing rather than nice-to-have.
- **The graph engine has subtler bugs available.** The checkpointer serializes
  state, so `snapshot.values["incident"]` is a *copy*. Approvals must be written
  back with `update_state`, and merging the copy back naively overwrites the event
  log. Both were real bugs during development, caught by tests. That hazard class
  simply does not exist in the explicit loop.
- The graph engine emits fewer distinct events (its routing is edge functions, not
  an LLM Supervisor), so the console shows slightly less detail.

**Revisit if:** the two implementations start needing genuinely different agent
behaviour. At that point they are no longer the same system and the parity test is
lying.
