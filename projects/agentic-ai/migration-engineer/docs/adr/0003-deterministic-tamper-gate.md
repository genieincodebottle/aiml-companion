# 0003. Detect reward hacking with git, not with a model's opinion

**Status:** Accepted

## Context

The agent is rewarded for making tests pass. The cheapest way to make a failing
test pass is to weaken it, and the very cheapest is to delete it. This is not a
hypothetical failure mode; it is the canonical one for coding agents, and an agent
that does it looks *perfect* on outcome metrics.

Something has to catch it, and that something cannot be the agent's own report.

## Options

1. **Trust the worker's `files_changed`.** Zero cost, and worthless: it is the
   agent's self-report about its own honesty.
2. **Ask a reviewer model** "does this diff look like cheating?" Flexible,
   catches subtle weakening (an assertion loosened rather than removed), but
   probabilistic - it can be argued out of a correct verdict, and it costs a call.
3. **Re-derive the truth from git**: did any file matching a test pattern change?
   Deterministic, cheap, and unarguable.

## Decision

Option 3 as the hard gate, with option 2 available on top. `harness/reviewer.py`
calls `changed_files(worktree)` - real `git diff --cached --name-only HEAD` - and
matches basenames against `_TEST_GLOBS` and paths against `_TEST_DIRS`. Any hit
sets `tampered_with_tests` and rejects, regardless of what the tests report. In
`live-sdk` mode an LLM critique can be layered on for richer prose reasons, but
the deterministic check always runs and always decides.

## Consequences

**Good**

- It cannot be talked out of its verdict, and it runs in both execution modes at
  no cost.
- Deletion is caught, not just modification - because `stage_all` runs
  `git add -A`, which stages removals, so the deleted path still appears in
  `--name-only`. That is a load-bearing detail of the implementation and it is now
  covered by `test_rejects_when_tests_were_deleted_outright`.
- Defence in depth: a deleted test also makes `run_tests` report failure, so the
  hack trips two independent gates.

**Bad**

- **It is a name check, not a semantics check.** An agent that weakens an assertion
  *inside* a file it was legitimately allowed to touch is invisible to it. The
  glob list is also fixed: a project whose tests are named `check_*.py` gets no
  protection at all, silently.
- False positives are possible and are not negotiable when they happen - a
  migration that genuinely requires a test update (say, a changed public API) is
  rejected, and the human has to override. That is the correct direction to err,
  but it does mean the gate blocks legitimate work.
- It only sees the final state of the worktree. An agent that deletes a test, runs
  the suite, and restores the file would pass.
