# 0002. Ship a stub worker that honours the identical contract

**Status:** Accepted

## Context

Every run of this system costs money and takes minutes, because a real coding agent
is doing real work. That is fatal for two things a learning project needs most: a
CI suite that runs on every push, and a reader who wants to see the system work
before deciding whether to spend anything.

## Options

1. **Mock the agent in tests only.** Cheap, but then CI exercises the mock's
   behaviour and nothing about the orchestration that ships.
2. **Record and replay real transcripts.** Faithful, but brittle - the recording
   goes stale the moment a prompt changes, and it cannot respond to a repo it has
   not seen.
3. **A deterministic stub worker behind the same interface** as the real ones.

## Decision

Option 3. `harness/stub_agent.py`, `harness/sdk_agent.py` (Claude Agent SDK) and
`harness/gemini_agent.py` all implement the same worker contract and are selected
by `config.execution_mode`. The stub performs the migration by applying the
rulebook's transformation directly - but it does so *through the same tool
registry*, so it is metered by the same budgets, filtered by the same guardrail
hooks, traced by the same observability, and reviewed by the same reviewer.

## Consequences

**Good**

- The full pipeline - fan-out, tools, hooks, budgets, review, PR - runs in CI for
  free and deterministically. That is what makes the golden eval possible at all.
- A reader clones and runs the whole thing with no key. The barrier to the first
  useful run is zero.
- The stub is a genuine implementation of the contract, so when it passes and the
  live worker fails, the difference is the model - not the harness.

**Bad**

- Two implementations of anything drift. Only discipline keeps `stub_agent.py`
  honest, and there is no test that proves it takes the same code path as
  `sdk_agent.py` - only that both satisfy the contract.
- **Stub results measure the harness, not the agent.** A green eval in stub mode
  says the orchestration works; it says nothing about whether a real model can do
  the migration. Reporting stub numbers as agent quality would be the single
  easiest way to mislead with this repo.
- Simulated cost is still cost-shaped. The stub charges a synthetic price per token
  (`_SIM_PRICE_PER_1M` in `tools/registry.py`) so the budget ceiling has something
  to meter. The scorecard now labels that figure "Simulated cost (USD)" - it did
  not always, and an unqualified `$0.031` reads exactly like a measurement.
