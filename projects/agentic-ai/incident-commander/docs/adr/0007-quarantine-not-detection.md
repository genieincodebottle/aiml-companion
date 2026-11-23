# 0007. Neutralize injected telemetry rather than flagging it

**Status:** Accepted

## Context

The original `sanitize_observations()` scanned telemetry for injection patterns,
appended any hits to a list, and returned **the observations unchanged**. The
poisoned text then reached the model verbatim.

The README described this as "prompt-injection defense". It was not. It was
logging. An attacker whose payload is delivered intact does not care that it was
also written to an audit trail.

## Options

1. **Detect and log only.** The status quo. Zero false-positive cost, zero
   protection.
2. **Detect and drop the whole field.** Strong, but destroys diagnostic signal - the
   poisoned log line usually sits next to the real error, and dropping it can remove
   the evidence needed to diagnose the actual fault.
3. **Redact, neutralize, and spotlight.** Surgical removal of the imperative text,
   everything else preserved and fenced.

## Decision

Option 3, in that order:

1. **Redact** secret- and PII-shaped substrings (`REDACTION_PATTERNS`).
2. **Neutralize** injection-shaped imperatives, substituting `QUARANTINE_MARKER`.
   The *fact* of the attempt survives for the audit trail; the instruction does not.
3. **Spotlight** the remainder inside `UNTRUSTED_TELEMETRY` fences carrying a
   standing instruction that content inside is data, never instructions.

Each layer covers a different gap: step 2 handles known patterns, step 3 limits
damage from unknown ones, and the architecture (closed enums + RBAC + human gate)
handles the case where both fail.

## Consequences

**Good**

- `test_injection_text_is_actually_removed_not_merely_flagged` asserts the payload
  is gone, so the claim is testable rather than rhetorical.
- Benign telemetry survives, so diagnostic quality is not sacrificed.
- Secrets in stack traces stop reaching the prompt, the postmortem, and whatever
  aggregates postmortems.
- `IC_QUARANTINE_INJECTIONS=false` restores flag-only mode, so a learner can run the
  poisoned scenario both ways and see the difference for themselves.

**Bad - and this needs saying loudly**

- **This does not solve prompt injection.** A regex list catches known phrasings.
  Novel ones get through, and anyone who claims otherwise is overselling. The real
  protection is architectural: a hijacked model can still only pick from a closed
  action menu, within its role's allowlist, with high-risk actions gated.
- **False positives are guaranteed.** The patterns are deliberately broad, so a
  legitimate log line containing "system prompt" gets a quarantine marker. That is
  the correct direction to err - a marker in the audit trail costs less than an
  unapproved production action - but it does degrade real telemetry sometimes.
- Sanitization walks every string in the observation tree on every pass. Fine at
  this scale, not free at a real one.
