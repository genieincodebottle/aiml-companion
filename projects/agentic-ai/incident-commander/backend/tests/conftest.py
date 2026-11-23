"""Shared test fixtures.

Two things happen here that every test depends on:

1. **The environment is pinned before `src` is imported.** `Settings` is a frozen
   dataclass memoized by `get_settings()`, so a test that sets an env var after
   import gets the stale value. Setting them at module import time - before any
   `src` import - is what makes the suite deterministic.

2. **Runbook memory is redirected to a temp file.** Without this, running the tests
   would write real episodes into `backend/data/runbook_memory.json`, and the suite
   would slowly change its own behaviour run over run. An agent with persistent
   memory needs test isolation for the memory too, which is easy to forget.

3. **The database and login are forced OFF.** `config.py` loads the developer's
   `backend/.env`, so a machine that happens to have `IC_DATABASE_URL` set for local
   work would otherwise run the whole suite against a real Postgres and behind an
   auth wall. That is not hypothetical - it turned 15 tests red the first time
   persistence was wired up, for reasons that had nothing to do with the code under
   test. These are overwritten rather than `setdefault`, because the entire point is
   to beat an inherited value.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# --- must happen before any `src` import ------------------------------------
_TMP_MEMORY = Path(tempfile.mkdtemp(prefix="ic-test-")) / "runbook_memory.json"
os.environ.setdefault("IC_MEMORY_PATH", str(_TMP_MEMORY))
# Both key spellings, FORCED rather than setdefault. A key exported in the
# developer's shell (or a monorepo-root .env) would otherwise survive into the
# suite, and any test that reaches the app's own `get_llm()` would call the real,
# paid API. Tests never call a real model.
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["IC_LLM_MODE"] = "auto"  # no key -> the offline agent
os.environ["IC_OFFLINE_STEP_DELAY_S"] = "0"  # the console delay would only slow the suite
os.environ.setdefault("IC_APPROVAL_TIMEOUT_S", "5")  # keep timeout tests quick
os.environ["IC_DATABASE_URL"] = ""  # the suite never touches a real database
os.environ["IC_AUTH_REQUIRED"] = "false"  # ...and is never behind a login

import pytest  # noqa: E402

from src.config import reset_settings_cache  # noqa: E402
from src.llm import reset_llm_cache, token_tracker  # noqa: E402
from src.loop import IncidentManager  # noqa: E402
from src.memory.runbook import get_memory, reset_memory_cache  # noqa: E402
from src.models.state import EventType, IncidentState  # noqa: E402

from ._fake_llm import FakeLLMClient  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip loop-engine-specific tests when the suite runs against the graph engine.

    Tests marked `loop_engine` assert the explicit loop's orchestration details -
    its event vocabulary, LLM-Supervisor routing, intermediate halt reasons. The
    graph engine reaches the same *outcomes* (that contract is
    `test_graph_parity.py`) but through conditional edges, so those assertions are
    meaningless there. Skipping here, rather than relying on everyone remembering
    `-m "not loop_engine"`, is what makes `IC_ENGINE=graph pytest -q` (the command
    the Makefile and CLAUDE.md actually advertise) run green.
    """
    if os.environ.get("IC_ENGINE", "loop").strip().lower() != "graph":
        return
    skip = pytest.mark.skip(reason="asserts explicit-loop orchestration; not meaningful under IC_ENGINE=graph")
    for item in items:
        if "loop_engine" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_memory():
    """Every test starts with an empty runbook memory and ends without leaking one."""
    reset_memory_cache()
    get_memory().clear()
    yield
    reset_memory_cache()


@pytest.fixture
def env(monkeypatch):
    """Set IC_* env vars and rebuild the settings cache.

    Returns a setter. Use it for anything that changes loop policy, e.g.
    `env(IC_MAX_ITERATIONS="1")`, and the caches are restored afterwards.
    """

    def _set(**kwargs: str):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, str(value))
        reset_settings_cache()
        reset_llm_cache()
        reset_memory_cache()

    yield _set
    reset_settings_cache()
    reset_llm_cache()
    reset_memory_cache()


@pytest.fixture
def manager() -> IncidentManager:
    """A manager wired to the deterministic test LLM."""
    return IncidentManager(llm=FakeLLMClient())


@pytest.fixture
def make_manager():
    """Build a manager with a specific stub, for the unhappy-path tests."""

    def _make(llm=None) -> IncidentManager:
        return IncidentManager(llm=llm if llm is not None else FakeLLMClient())

    return _make


async def drive(
    manager: IncidentManager,
    scenario_id: str,
    decision: str | None = "approve",
    *,
    auto_approve: bool = False,
    note: str = "test",
    on_event=None,
) -> IncidentState:
    """Run an incident to completion, optionally answering the approval gate.

    `decision=None` leaves the gate unanswered, which is how the approval-timeout
    path is exercised.
    """
    state = manager.trigger(scenario_id, None, auto_approve=auto_approve)
    async for ev in manager.subscribe(state.incident_id):
        if on_event is not None:
            on_event(ev, manager, state)
        if ev.type == EventType.AWAITING_APPROVAL and decision is not None:
            manager.resolve_approval(state.incident_id, ev.payload["action_id"], decision, note)
    return manager.get(state.incident_id)


@pytest.fixture
def drive_incident():
    """The `drive` helper, as a fixture, for readability in tests."""
    return drive


@pytest.fixture(autouse=True)
def clean_token_ledger():
    yield
    token_tracker._by_incident.clear()  # noqa: SLF001 - test cleanup of a module singleton
