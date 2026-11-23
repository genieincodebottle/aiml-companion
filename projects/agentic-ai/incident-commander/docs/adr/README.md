# Architecture Decision Records

Short records of decisions that had a **real alternative**. Each one states the
context, the options considered, what was chosen, and - most importantly - what it
costs.

A decision with no downside is not a decision, it is a default. These are the
places where something was genuinely traded away, and knowing what was given up is
the difference between understanding a system and memorizing it.

| # | Decision | Status |
|---|---|---|
| [0001](0001-explicit-loop-and-langgraph.md) | Ship the agent loop twice: explicit and as a StateGraph | Accepted |
| [0002](0002-llm-routes-code-governs.md) | The model decides what to do; code decides what is allowed | Accepted |
| [0003](0003-closed-enums-over-free-text.md) | Bind agent outputs to closed enums, not strings | Accepted |
| [0004](0004-falsifiable-predictions.md) | Every proposal must make a checkable prediction | Accepted |
| [0005](0005-lexical-memory-over-embeddings.md) | Lexical (BM25) retrieval for runbook memory, not embeddings | Accepted |
| [0006](0006-negative-scenarios-in-eval.md) | Score "escalated correctly" as strictly as "resolved" | Accepted |
| [0007](0007-quarantine-not-detection.md) | Neutralize injected telemetry rather than flagging it | Accepted |
| [0008](0008-adversarial-critic.md) | A second agent argues against every proposal | Accepted |
| [0009](0009-gemini-mandatory-no-fallback.md) | No silent fallback when the model is unavailable; a labelled offline mode instead | Amended |
| [0010](0010-integrating-real-infrastructure.md) | Integrate real infrastructure behind protocols, and earn autonomy in tiers | Accepted (design) |

## Format

Kept deliberately short. If an ADR needs more than a page, the decision probably
needs splitting.

```
# NNNN. Title
Status | Context | Options | Decision | Consequences (including the bad ones)
```
