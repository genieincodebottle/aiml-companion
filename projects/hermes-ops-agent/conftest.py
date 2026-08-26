"""Make the test suite work on a fresh clone, from any working directory.

Two things a beginner should never have to know about:

1. The fixture .db is gitignored (the .sql is the source of truth), so a fresh
   clone has none. Six tests used to fail with no explanation. It is built
   automatically here instead.
2. `from src...` imports need the project root on sys.path. Running
   `pytest` from the repo root rather than from this directory used to be a
   coin flip.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import fixture  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _built_fixture():
    """Build fixtures/state.fixture.db before anything runs."""
    fixture.ensure()
