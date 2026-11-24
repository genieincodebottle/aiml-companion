# 0001. Every model call goes through one gateway

**Status:** Accepted

## Context

A control plane governs agents. Governing them means being able to say, for every
model call in the system: which model served it, what it cost, who asked, what was
redacted before it left, and whether the answer came from cache.

If agents call providers directly, none of that is knowable without instrumenting
every agent - and every new agent is a new hole.

## Options

1. **Agents call providers directly**, with a shared helper by convention. Simple,
   and convention is not enforcement: the first agent that imports the SDK
   directly is invisible to the platform.
2. **A sidecar proxy.** Genuinely unavoidable, language-agnostic, and now the
   platform has an operational component to deploy, scale and debug.
3. **One in-process gateway** that every agent must call.

## Decision

Option 3. `gateway/core.py` is the single entry point; `gateway/providers.py` is
the only place a provider SDK is touched. Routing by `ModelTier` (premium/cheap),
budgets, redaction, the semantic cache and trace emission all live on that path,
so they apply to every call by construction rather than by discipline.

## Consequences

**Good**

- Swapping providers, adding a tier, or changing the redaction policy is one file.
  The claim "the seam is one class" is literally true and worth checking for
  yourself.
- Cost, cache-hit rate and model attribution are complete rather than
  best-effort - there is no path that spends without being counted.
- Stub and live modes differ only in which provider the gateway holds, which is
  what makes [ADR 0006](0006-stub-provider-drives-every-plane.md) possible.

**Bad**

- In-process means same-language and same-deployment. A JavaScript agent, or one
  running in someone else's container, cannot be governed by this gateway - it
  would need option 2, and the whole design would change shape.
- It is a single point of failure and a bottleneck. Nothing here is concurrent-safe
  by accident; the budget counters and cache are shared mutable state.
- **Enforcement is architectural, not technical.** Nothing stops an agent importing
  a provider SDK directly and bypassing the gateway entirely. In this codebase that
  is caught by review; at scale it needs a lint rule or a network policy.
