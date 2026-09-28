# 0001. Edit real git worktrees, not a simulated filesystem

**Status:** Accepted

## Context

The agent's job is to change source files across a fleet of repositories. It needs
somewhere to make those changes, and something to produce a diff from.

The tempting shortcut is an in-memory filesystem: a dict of path -> contents that
the agent edits, with a hand-written differ at the end. It is fast, has no external
dependency, and is trivially isolated per job.

## Options

1. **Simulated filesystem.** Fast, hermetic, and easy to reset between runs.
2. **A shared checkout the agent edits in place.** Simple, but concurrent repos
   collide and a failed run leaves the working tree dirty for the next one.
3. **A real `git worktree` per repo per job**, diffed with real `git diff`.

## Decision

Option 3. `src/worktree.py` and `src/vcs/local.py` check out an isolated worktree
under `.work/`, the worker edits real files on disk, and `harness/diffing.py`
produces the diff by staging and calling `git diff --cached HEAD`.

## Consequences

**Good**

- The diff is not an approximation of what git would say - it *is* what git says,
  so what the reviewer inspects and what a PR would contain cannot disagree.
- Isolation is per worktree, so the fleet can run repos concurrently without a
  shared mutable state to reason about.
- The tools the agent calls (`read_file`, `write_file`, `run_tests`) are the same
  ones that would work against a real checkout. There is no "now port it to real
  files" step waiting at the end.
- It makes a whole class of defence possible that a simulation cannot: see
  [0003](0003-deterministic-tamper-gate.md). `git add -A` stages deletions, which
  is the only reason deleting a test file is detectable at all.

**Bad**

- Slower. The test suite spends most of its wall clock on git and subprocesses -
  ~27s of the ~40s run, versus milliseconds for a simulation.
- Requires git on PATH, so the project is no longer pure-Python-and-nothing-else.
- `.work/` accumulates worktrees. It is gitignored, but it is real disk: the
  fixture fleet leaves ~5MB behind after an eval run, and nothing prunes it
  automatically.
- Windows path and encoding handling had to be dealt with explicitly (see the
  UTF-8 note in `tools/testrunner.py`), which a simulation would have avoided.
