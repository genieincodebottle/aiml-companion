# 0008. Set configuration before importing anything that memoizes it

**Status:** Accepted

## Context

`config.get_settings()` memoizes a frozen `Settings` built from environment
variables. That is the right shape - configuration read once, immutable
thereafter - but it creates a sharp ordering rule: whatever reads the environment
first wins, forever.

Tests must run against the deterministic stub worker with an isolated data
directory. If any `src` module is imported before those variables are set, the
memoized settings capture the developer's ambient environment instead.

## Options

1. **A pytest fixture that sets the env vars.** The obvious approach, and it runs
   *after* module import, so it is too late for anything captured at import time.
2. **Pass settings explicitly everywhere.** Correct and invasive; every call site
   grows a parameter.
3. **Set the variables at the top of `conftest.py`, before importing `src`**, and
   expose a `reset_settings_cache()` hook.

## Decision

Option 3. `tests/conftest.py` sets `ME_FORCE_STUB` and `ME_DATA_DIR` as its first
statements, with `from src.config import reset_settings_cache` deliberately below
them and a `# noqa: E402` acknowledging that the import order is the point.

## Consequences

**Good**

- Tests never touch the network or a real SDK, whatever the developer has exported.
- The ordering constraint is stated in the file where it matters, rather than being
  folklore that survives only as long as the person who knew it.

**Bad**

- **It is enforced by a comment, not by the language.** Nothing stops a future
  import creeping above those lines, and the failure would be silent and
  environment-dependent - green on CI, live and billed on a developer machine that
  happens to have a key exported.
- Module-level side effects in `conftest.py` are unusual enough to look like a
  mistake to someone tidying imports, which is exactly the person most likely to
  break it.

**Worth knowing:** the sibling project `mission-control` got this wrong in the
other direction, and it is the clearest illustration of why this ADR exists. Its
`conftest.py` used a fixture (option 1) while its `main.py` resolved settings and
built its control plane at import time. On a machine with an API key exported, its
"offline, no network, fully reproducible" suite therefore ran against live
Gemini - 84.4s and a failing assertion, versus 1.1s and green with the key
unset. Same class of bug, opposite file, and the only visible symptom was one test
failing in a way that looked like a broken assertion. Both are now fixed;
mission-control's [ADR 0008](../../../mission-control/docs/adr/0008-resolve-settings-per-request.md)
records the other half of the lesson.
