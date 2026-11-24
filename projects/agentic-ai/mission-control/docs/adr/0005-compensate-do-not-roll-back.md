# 0005. Compensate with sagas; there is no distributed transaction

**Status:** Accepted

## Context

Committing a disposition is not one write. It records a case, notifies a downstream
system, and updates an index - across a database, a message bus and an external
service. If the third step fails, the first two have already happened.

There is no transaction that spans a database, a message bus and an LLM. Pretending
otherwise is how systems end up with cases that exist in one place and not another.

## Options

1. **Do nothing and hope.** Partial commits accumulate silently, and the
   inconsistency is discovered weeks later by a human reconciling reports.
2. **Two-phase commit.** Real transactional semantics, and it requires every
   participant to support it. An external notification API does not.
3. **Saga with compensating actions**: record how to undo each step as it
   completes, and unwind in reverse on failure.

## Decision

Option 3. `runtime/saga.py` accumulates `(name, compensate)` pairs as forward steps
succeed and runs the compensations in reverse order when a later step raises,
wrapping the cause in `SagaError`. Work that cannot be compensated at all goes to
the dead-letter queue (`runtime/dlq.py`) for a human, rather than being retried
into a worse state.

## Consequences

**Good**

- Failure leaves a consistent world or an explicit dead letter - never a silent
  partial commit.
- Compensation is written next to the forward action, at the moment the author
  knows how to undo it, rather than in a recovery script written a year later by
  someone who does not.
- The DLQ makes "we could not fix this automatically" a visible state with an owner.

**Bad**

- **Compensation is not rollback.** Un-sending a notification is sending a second
  notification. The downstream system saw the first one, and any side effect it
  triggered has already happened. The world is reconciled, not restored.
- A compensation can itself fail, and then you are in the state the saga existed to
  prevent, with less information about how you got there.
- Every forward action now needs an inverse, which is real design work and is
  sometimes genuinely impossible - "un-file a regulatory report" is not a thing.
  Those actions belong behind the human gate ([ADR 0002](0002-fail-closed-policy-engine.md)),
  not behind a compensation.
- Sagas are harder to reason about than transactions, and the intermediate states
  are visible to anyone reading the database at the wrong moment.
