# Architecture Decision Records

Short records of decisions that had a **real alternative**. Each one states the
context, the options considered, what was chosen, and - most importantly - what it
costs.

A decision with no downside is not a decision, it is a default. These are the
places where something was genuinely traded away, and knowing what was given up is
the difference between understanding a system and memorizing it.

Mission Control is a **control plane**, not an agent. Most of these decisions are
about governing agents you did not write and cannot fully trust.

| # | Decision | Status |
|---|---|---|
| [0001](0001-one-gateway-seam.md) | Every model call goes through one gateway | Accepted |
| [0002](0002-fail-closed-policy-engine.md) | Policy is data, evaluated per call, failing closed | Accepted |
| [0003](0003-stable-redaction-placeholders.md) | Redact to stable placeholders, not to `[REDACTED]` | Accepted |
| [0004](0004-real-vectors-in-the-stub-cache.md) | A real vector in the semantic cache, even offline | Accepted |
| [0005](0005-compensate-do-not-roll-back.md) | Compensate with sagas; there is no distributed transaction | Accepted |
| [0006](0006-stub-provider-drives-every-plane.md) | The stub provider drives every plane, not a mock | Accepted |
| [0007](0007-a-judge-that-can-fail.md) | The promotion gate needs a judge that can return a bad score | Accepted |
| [0008](0008-resolve-settings-per-request.md) | Resolve settings and singletons per request, never at import | Accepted |
| [0009](0009-maker-checker-and-durable-decisions.md) | Maker-checker, and decisions that survive a restart | Accepted |
| [0010](0010-promote-only-on-a-live-measurement.md) | Promote a prompt only on a live measurement | Accepted |

## Format

Kept deliberately short. If an ADR needs more than a page, the decision probably
needs splitting.

```
# NNNN. Title
Status | Context | Options | Decision | Consequences (including the bad ones)
```
