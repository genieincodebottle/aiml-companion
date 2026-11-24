# 0004. A real vector in the semantic cache, even offline

**Status:** Accepted

## Context

Two prompts that mean the same thing should not both be paid for. That requires
similarity, not equality - and similarity requires an embedding, which normally
requires a model, a key and a network.

The platform's core promise is that everything runs offline. A cache that only
works in live mode would be a plane the offline demo cannot exercise.

## Options

1. **Exact-match cache on a hash of the prompt.** Trivial, offline, and misses the
   entire point: near-duplicates are the common case and would all miss.
2. **Fake the similarity offline** - return a hit if the strings look close, e.g.
   by edit distance or a token-overlap ratio. Cheap, and the *behaviour* would be a
   different algorithm from the one that ships, so the offline demo would teach the
   wrong mental model.
3. **A deterministic hashed bag-of-words embedding**, with genuine cosine
   similarity and a threshold.

## Decision

Option 3. `gateway/cache.py` embeds each prompt into a 256-dimensional
L2-normalized vector by hashing tokens into buckets, then does real cosine
similarity against stored entries. No network, no key, fully deterministic - and
the *mechanism* is the one that ships. Swapping `_embed` for a real embedding model
changes nothing else.

## Consequences

**Good**

- Near-duplicate hits and novel-prompt misses are real, observable behaviour
  offline, so a learner can see a semantic cache work rather than read about one.
- The seam for a production embedding is a single function.
- Threshold tuning - the actual hard part of semantic caching - can be explored
  with no cost and no rate limit.

**Bad**

- **Hashed bag-of-words has no semantics.** "transfer approved" and "transfer
  denied" share most of their tokens and will score as near-identical; a real
  embedding would separate them. So the cache demonstrates the *plumbing*
  faithfully and the *quality* not at all, and a learner who does not notice may
  conclude semantic caching is more reliable than it is.
- That failure mode is the dangerous direction: a false cache hit returns a
  confidently wrong answer that never reaches the model.
- The threshold that looks right for hashed bag-of-words is not the threshold that
  will be right for a real embedding model. It has to be re-tuned on the swap, and
  nothing forces that.
- Cache entries are per-process and unbounded. No eviction, no persistence, no
  sharing across replicas.
