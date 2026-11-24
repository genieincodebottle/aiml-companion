# 0006. The stub provider drives every plane, not a mock

**Status:** Accepted

## Context

The platform has six planes: gateway, policy, runtime, HITL, evals, tracing. All of
them need to be exercisable by someone who has just cloned the repo, and by CI on
every push. None of that can depend on a paid API key.

## Options

1. **Mock the model in tests.** Then CI verifies the mock's behaviour, and the
   orchestration that actually ships is exercised only by hand, occasionally,
   by whoever has a key.
2. **Record and replay real responses.** Faithful to a real model's output, and
   stale the moment a prompt changes - the recording cannot answer a question it
   was not recorded for.
3. **A deterministic in-process provider** that produces schema-valid outputs
   derived from the prompt's signals, plugged in at the gateway seam.

## Decision

Option 3. `tenants/fincrime/stub_brain.py` synthesizes a valid instance of whatever
Pydantic schema was requested, keyed off the *data* in the user message and
deliberately never off the system instructions - so an instruction word like
"unsupported" cannot be mistaken for evidence. It is selected in
`gateway/providers.py`, which means every plane above it - policy, runtime, saga,
HITL, evals, tracing - runs the identical code path it runs in live mode.

## Consequences

**Good**

- The entire platform runs offline, free, deterministically, in CI. That is what
  makes the golden eval a gate rather than a demo.
- The stub is a *faithful stand-in*, not a mock: it goes through the gateway, gets
  budgeted, redacted, cached and traced like any other call.
- Determinism means the golden expectations hold exactly, so a regression is a
  real signal rather than noise.

**Bad**

- **The stub is not a model, and stub results are not agent quality.** Green in
  stub mode means the orchestration works. It says nothing about whether a real
  model would reach the same disposition, and reporting stub numbers as if they
  were agent performance would be the easiest way to mislead with this repo.
- Signal-keying makes the stub's behaviour a function of prompt *format*. Change
  how an agent phrases its evidence and the stub's answer changes, which looks
  like a model regression and is not.
- It bounds what the offline eval can catch. The judge in stub mode comes from the
  same brain as the subject, so it is a consistency check, never an independent
  opinion - see [ADR 0007](0007-a-judge-that-can-fail.md), which is exactly where
  that assumption went wrong once already.
