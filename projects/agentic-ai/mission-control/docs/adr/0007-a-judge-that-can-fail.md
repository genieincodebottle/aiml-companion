# 0007. The promotion gate needs a judge that can return a bad score

**Status:** Accepted

## Context

The promotion gate is `accuracy >= 1.0 AND mean_judge_score >= MC_EVAL_THRESHOLD`.
Programmatic accuracy answers "did it reach the right disposition?". The judge is
supposed to answer the harder question: "is the narrative actually grounded in
the evidence, or does it just sound like it is?"

For a long time it could not answer that, and nobody noticed - because a gate that
always passes looks exactly like a gate that is never triggered.

The stub judge decided `grounded` by scanning the narrative prose:

```python
grounded = not _has(text, "uncited", "unsupported", "grounded=false")
```

That is a confession detector. A narrative is ungrounded because it *lacks*
citations, not because it announces that it lacks them. Measured:

| narrative | old score |
|---|---|
| 2 claims, 0 evidence items | **1.00** |
| 1 claim, 0 evidence items | **1.00** |
| 1 claim containing the word "uncited" | 0.60 |

So in stub mode - the default, and what CI runs - the judge term was effectively
constant, and the gate collapsed to its accuracy term.

## Options

1. **Drop the judge offline** and gate on accuracy alone. Honest, and it removes
   the grounding check entirely from CI.
2. **Require a live model for the eval.** A real second opinion, and now the
   promotion gate costs money and needs a key, so it stops running on every push.
3. **Give the judge structural signals it can actually check**, computed by the
   eval rather than inferred from the subject's own prose.

## Decision

Option 3, following the convention the critic agent already used - it passes
`every_claim_cited=` as an explicit signal rather than hoping the model infers it.
`evaluation/run_eval.py` now computes `_citation_gap()` and passes
`uncited_claims=N evidence_count=M`, and the stub judge scores from those. A caller
that supplies no structural signal gets a verdict capped below the promotion
threshold, so an unchecked run can never clear the gate.

## Consequences

**Good**

- The judge term can now fail: ungrounded narratives score 0.6 against a threshold
  of 0.8. The gate has two live terms instead of one.
- The three golden alerts still score 1.00 - they genuinely are grounded - so the
  check discriminates rather than merely being pessimistic. A judge that fails
  everything is as useless as one that passes everything; it just fails in the
  safe direction, which is how a broken gate survives review.

**Bad**

- **The structural check is weak.** `EvidenceItem` carries no claim id, so "cited"
  can only mean "there are at least as many evidence items as claims". A narrative
  with three claims and three *irrelevant* evidence items passes. A real citation
  graph is the right fix and is not built.
- Computing the signal in the eval means the eval and the judge share an
  assumption. If `_citation_gap` is wrong, the judge is wrong in the same
  direction, and nothing catches it.
- **In stub mode the judge is not independent at all.** It is the same
  deterministic brain as the subject, so it is a consistency check. In live mode
  it is a separate call, but still a same-family model, and self-preference is
  well documented for LLM judges. Neither is the "independent judge" the docs used
  to claim.
