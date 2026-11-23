# 0005. Lexical (BM25) retrieval for runbook memory, not embeddings

**Status:** Accepted

## Context

Runbook memory needs to retrieve past incidents similar to the current one. The
default instinct is embeddings plus a vector store, because that is what "semantic
search" means in most RAG tutorials.

## Options

1. **Embeddings + a vector DB (Chroma/pgvector).** Handles paraphrase. Adds a
   dependency, an API call per write and per read, a cold start, and
   non-determinism that makes retrieval untestable.
2. **Lexical BM25 over symptom text.** Standard library only, deterministic,
   inspectable. Misses paraphrase.
3. **Hybrid.** Best quality, most machinery.

## Decision

Option 2, with the seam for option 3 left open.

The reasoning is specific to this domain rather than a general preference:

- **Incident telemetry is near-identifier text.** `OutOfMemoryError`,
  `HikariPool-1`, `SSLHandshakeException`, `PKIX path validation`. Lexical matching
  is *strong* on identifiers, and embeddings are comparatively weak on them - the
  classic failure being that two different exception classes score 0.94 similar
  because both are "a Java error". That is precisely the confusion that would make
  a runbook recall actively harmful.
- **Deterministic retrieval is testable.** `tests/test_memory.py` asserts which
  episode comes back and which terms matched. That is not possible with a hosted
  embedding model.
- **Inspectable.** Every hit reports `matched_terms`, so "why did it recall that?"
  has an answer a learner can read.
- **`min_score` suppresses the retrieve-something-anyway failure**, which matters
  more than recall here: an irrelevant past incident anchors the model on the wrong
  cause, and is worse than no memory at all.

## Consequences

**Good**

- Zero dependencies, zero cost, no cold start, fully deterministic.
- The whole implementation is around 40 lines and readable, so memory is not a
  black box in a project about making agent behaviour legible.
- `data/runbook_memory.json` can be opened and read: you can see exactly what the
  agent remembers.

**Bad**

- **Paraphrase is missed.** "heap exhausted" will not match "out of memory". This
  is a real quality ceiling, stated in the module docstring rather than hidden.
- BM25 parameters (`k1`, `b`) are untuned defaults; nobody has checked they suit
  short symptom documents.
- The stopword list is hand-written and short. Over-aggressive stopwords would drop
  `pool` or `timeout`, which carry real signal here - so it errs small, at the cost
  of some noise.

**Revisit if:** symptom text starts coming from human descriptions rather than
machine telemetry. Humans paraphrase; machines do not. At that point hybrid search
stops being an exercise and becomes necessary.
