# 0003. Redact to stable placeholders, not to `[REDACTED]`

**Status:** Accepted

## Context

Financial-crime investigations are made of exactly the data you must not send to a
third-party model: names, account numbers, emails. But the *reasoning* depends on
relationships between those entities - "the account that received the first
transfer also received the third" is the finding.

Redact naively and you destroy the reasoning while protecting the data.

## Options

1. **Send it raw.** The model reasons perfectly and the PII leaves your boundary.
   Not available in this domain.
2. **Replace every match with `[REDACTED]`.** Safe, and it collapses distinct
   entities into one token: `[REDACTED]` paid `[REDACTED]` is unanalysable, and
   worse, a model may hallucinate a relationship to fill the gap.
3. **Replace with stable typed placeholders** - `[EMAIL_1]`, `[ACCOUNT_2]` -
   consistent within a call, with the mapping retained for re-hydration.

## Decision

Option 3. `gateway/redaction.py` assigns each distinct value a numbered placeholder
by type, reuses it for every later occurrence in the same call (so co-reference
survives), and returns the mapping so the caller can re-hydrate the response.

## Consequences

**Good**

- The model can still reason about "the same account appearing twice", which is
  most of what this domain needs.
- Typed placeholders keep the shape of the data visible, so the model knows an
  account number is an account number.
- Re-hydration means the human sees real values in the final narrative while the
  provider never did.

**Bad**

- **Placeholders are stable within a call, not across calls.** `[ACCOUNT_1]` in
  step 3 need not be `[ACCOUNT_1]` in step 5, so cross-step reasoning about the
  same entity is not preserved. Making them stable across a workflow means keeping
  a per-workflow mapping, which is a longer-lived store of exactly the data you
  were trying not to keep.
- Regex-based detection has the usual two failure modes: it misses PII that does
  not match a pattern (a name is not a pattern), and it mangles legitimate text
  that does.
- The mapping itself is sensitive. It lives in memory for the duration of a call,
  and anything that logs or serializes it re-creates the leak in a new place.
- Re-hydration can reintroduce PII into a stored artifact if a caller re-hydrates
  before persisting rather than after displaying.
