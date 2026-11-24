# 0008. Resolve settings and singletons per request, never at import

**Status:** Accepted

## Context

`config.get_settings()` and `control_plane.get_control_plane()` both memoize. That
is correct: configuration should be read once, and the plane is expensive to build.

`main.py` then did this at module scope:

```python
settings = get_settings()
...
plane = get_control_plane()
```

Both run at **import** time, which is before any pytest fixture. `tests/conftest.py`
sets `MC_FORCE_STUB=1` in an autouse fixture and promises "no key, no network,
fully reproducible" - but by the time that fixture ran, `import main` had already
snapshotted the ambient environment and built a control plane bound to whatever
provider it implied.

On a developer machine with `GEMINI_API_KEY` exported, the offline suite ran
against live Gemini:

| | result |
|---|---|
| `tests/test_api.py`, key present | 84.4s, 1 failed (`mode == "live"`) |
| `tests/test_api.py`, key unset | 1.1s, 4 passed |
| full suite after the fix, key still present | 2.6s, 51 passed |

The 75x gap was real billed traffic. The only visible symptom was
`test_health_reports_stub_mode` failing, which reads like a broken assertion rather
than what it was: the suite honestly reporting that it was not offline.

## Options

1. **Set the env vars at the top of `conftest.py`, before importing `src`.** This
   is what the sibling project `migration-engineer` does
   ([its ADR 0008](../../../migration-engineer/docs/adr/0008-configure-before-import.md)),
   and it works - but it makes import order load-bearing and enforces it with a
   comment.
2. **FastAPI dependency injection** for settings and the plane. Idiomatic, and it
   changes every handler signature.
3. **Resolve through accessor functions inside each handler.**

## Decision

Option 3, plus defence in depth. `main.py` keeps a single `_boot = get_settings()`
for the two things that genuinely must exist before the first request - log level
and CORS origins - and every handler calls `_settings()` / `_plane()`. Both already
memoize, so the cost is a dict lookup, but the resolution happens *after* the
environment is settled.

`reset_control_plane()` was added alongside `reset_settings_cache()`, because
resetting settings alone is not enough: the plane binds a provider when it is
constructed, so a plane built live stays live however the settings are reset
afterwards. The conftest fixture now deletes `GOOGLE_API_KEY` and `GEMINI_API_KEY`
outright as well - if a future refactor reintroduces an import-time snapshot, the
force-stub switch is read too late to help, but a missing key still cannot become a
live provider.

## Consequences

**Good**

- The suite is genuinely offline regardless of what the developer has exported, and
  `test_offline_guarantee.py` fails if that stops being true.
- Config becomes overridable at runtime - by tests, a CLI flag, or a future
  per-tenant override - rather than being frozen by whoever started the process.
- An import-time singleton is not configuration; it is a snapshot of whoever
  happened to start the process. Naming the surviving one `_boot` says so.

**Bad**

- Two accessor functions that look like indirection for its own sake, and the
  reason is a comment. Someone will inline them.
- `_boot` is still an import-time snapshot for CORS and logging, so the class of
  bug is reduced, not eliminated - a future setting that belongs in `_boot` but is
  environment-sensitive would reintroduce it.
- Per-request resolution makes it possible for settings to change mid-run, which is
  a new kind of inconsistency that could not previously occur.
