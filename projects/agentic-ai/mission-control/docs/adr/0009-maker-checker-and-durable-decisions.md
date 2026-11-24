# 0009. Maker-checker, and decisions that survive a restart

**Status:** Accepted

## Context

A case disposition is a regulatory filing. Two things were wrong with how a person
decided one:

1. Anyone who could reach the API could approve, and nothing recorded who did. The
   approve endpoint took no identity at all.
2. A decision only worked while the case was in memory. After a restart it raised
   "restart or replay required", although the README promised durable execution and
   every step was checkpointed.

## Options

1. Accounts in a database with passwords and JWTs.
2. Operator tokens from configuration, plus maker-checker enforced in the runtime.
3. Leave identity to a reverse proxy and trust a header.

## Decision

Option 2. `MC_OPERATORS` lists people, their roles (analyst, reviewer) and a token
each. With it unset the console sends an unverified name, which is honest for a
laptop and says so on screen. Either way the **runtime**, not the API, refuses a
decision from the person who opened the case: four eyes on every filing, whatever
roles that person holds. A decision on a case not in memory rebuilds it from its
checkpoint and event log, and a per-case lock makes a double click harmless.

## Consequences

- Who opened and who decided a case are on the record, in the API and the console.
  `policy:auto_approve` is recorded as itself, never as a person.
- Tokens in an environment variable do not rotate or expire. That is acceptable for
  a team-sized deployment and is the first thing to replace (SSO) for anything
  larger; the `operator` function in `main.py` is the only place it lives.
- Rebuilding a case needs every tenant registered in `WORKLOADS`, and every
  in-process object recreated by `Workload.rehydrate`. A tenant that forgets the
  second will resume with a missing tool and dead-letter on its next step.
- The checkpoint format is versioned. A file with an unknown version is refused, not
  guessed at, so changing the layout means bumping `CHECKPOINT_VERSION` and saying
  what happens to parked cases.
