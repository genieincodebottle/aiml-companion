"""
Regression tests for the offline guarantee.

The bug
-------
`tests/conftest.py` promises: "Every test runs in stub mode against a temp data
dir - no key, no network, fully reproducible." It sets `MC_FORCE_STUB=1` in an
autouse fixture.

`main.py` defeated that with two module-level lines::

    settings = get_settings()
    ...
    plane = get_control_plane()

Both run at IMPORT time, which is before any fixture. `import main` at
collection therefore snapshotted the ambient environment and built a control
plane bound to whatever provider that environment implied. On a developer
machine with `GEMINI_API_KEY` exported, the "offline" API suite ran against
live Gemini. Measured:

    tests/test_api.py, key present : 84.4s, 1 failed (mode == "live")
    tests/test_api.py, key unset   :  1.1s, 4 passed

    full suite after the fix, key still present : 2.6s, 31 passed

That 75x gap was real billed traffic, and the one visible symptom -- a failing
`test_health_reports_stub_mode` -- looked like a broken assertion rather than
what it was: the suite reporting honestly that it was not offline.

The fix resolves settings and the plane per request, and the fixture now also
deletes the keys and resets both singletons. These tests fail against the old
`main.py` on any machine with a key configured.

Run: pytest tests/test_offline_guarantee.py -v
"""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

import main
from src.config import get_settings
from src.control_plane import get_control_plane, reset_control_plane

# === The guarantee itself ===


def test_the_app_is_offline_even_when_a_key_is_exported(monkeypatch):
    """The decisive test. The old code reported "live" here."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-looks-real-enough")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-looks-real-enough")
    from src.config import reset_settings_cache

    reset_settings_cache()
    reset_control_plane()

    body = TestClient(main.app).get("/api/health").json()
    assert body["mode"] == "offline"
    assert body["premium_model"] is None
    assert body["cheap_model"] is None


def test_no_live_provider_is_ever_constructed_during_tests():
    settings = get_settings()
    assert settings.mode == "offline"
    assert settings.use_live_provider is False
    assert settings.execution_mode == "offline"


def test_the_api_keys_are_absent_inside_the_test_process():
    """Belt as well as braces: even if the force-stub switch were read too
    late by some future refactor, there is no key for it to find."""
    import os

    assert not os.getenv("GOOGLE_API_KEY")
    assert not os.getenv("GEMINI_API_KEY")


# === No import-time snapshots ===


def test_main_does_not_bind_settings_or_the_plane_at_import():
    """Guards against reintroducing the exact two lines that caused this.

    An import-time singleton is not configuration -- it is a snapshot of
    whoever happened to start the process, taken before any test fixture,
    environment override, or CLI flag can have an opinion.
    """
    source = inspect.getsource(main)
    body = source.split("# --- request models ---")[0]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"'):
            continue
        assert not stripped.startswith("settings = get_settings()"), (
            "main.py must not snapshot settings at import; resolve per request"
        )
        assert not stripped.startswith("plane = get_control_plane()"), (
            "main.py must not build the control plane at import"
        )


def test_handlers_resolve_the_plane_through_the_accessor():
    source = inspect.getsource(main)
    handlers = source.split("# --- request models ---")[1]
    assert "_plane()" in handlers
    assert "_settings()" in handlers


# === The reset hooks must actually reset ===


def test_reset_control_plane_forces_a_rebuild():
    """Resetting settings alone is not enough: the plane binds its provider at
    construction, so a plane built live stays live however settings change."""
    first = get_control_plane()
    assert get_control_plane() is first, "should be memoized"
    reset_control_plane()
    assert get_control_plane() is not first, "should rebuild after reset"


def test_each_test_gets_an_isolated_data_dir(settings, tmp_path):
    assert settings.data_dir == tmp_path


# === And the suite stays fast, which is the observable symptom ===


def test_the_health_check_does_not_touch_the_network():
    """A live-mode health check took ~1s of round trip; stub mode is local.
    This is a smoke check on the symptom, not a substitute for the assertions
    above -- a slow machine could make it flaky, so the bound is generous."""
    import time

    client = TestClient(main.app)
    start = time.perf_counter()
    for _ in range(5):
        client.get("/api/health")
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"5 health checks took {elapsed:.1f}s -- is this live?"
