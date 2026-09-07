"""A fresh clone must work with no setup step.

The fixture .db is gitignored, so a newly cloned repo has none. Before this,
`python main.py demo` printed "Run: make fixture" and stopped - on Windows,
where `make` is usually absent, that was a dead end on the first command in
the README. Six tests also failed for the same reason.

These tests delete the fixture and assert the entry points rebuild it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from src import fixture

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def no_fixture_db():
    """Simulate a fresh clone, then restore whatever was there."""
    db = fixture.DB_PATH
    backup = db.read_bytes() if db.exists() else None
    if db.exists():
        db.unlink()
    yield db
    if backup is not None:
        db.write_bytes(backup)
    else:
        fixture.ensure()


class TestFreshClone:
    def test_demo_builds_the_fixture_itself(self, no_fixture_db):
        r = subprocess.run(
            [sys.executable, "main.py", "demo"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert "skill shortened the procedure" in r.stdout
        assert no_fixture_db.exists()

    def test_no_error_message_tells_the_user_to_run_make(self, no_fixture_db):
        # `make` is not installed on a default Windows box. An error whose only
        # remedy needs make is not a remedy.
        r = subprocess.run(
            [sys.executable, "main.py", "demo"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert "make fixture" not in (r.stdout + r.stderr)

    def test_ensure_is_idempotent(self):
        first = fixture.ensure()
        stamp = first.stat().st_mtime_ns
        again = fixture.ensure()
        assert again == first
        assert again.stat().st_mtime_ns == stamp, "ensure() rebuilt an existing fixture"

    def test_ensure_rebuilds_a_zero_byte_fixture(self, no_fixture_db):
        # A truncated artefact is worse than a missing one: SQLite opens it as
        # a valid empty database.
        no_fixture_db.write_bytes(b"")
        fixture.ensure()
        assert no_fixture_db.stat().st_size > 0

    def test_build_reports_a_missing_sql_source_clearly(self, tmp_path):
        with pytest.raises(fixture.FixtureError, match="clone is incomplete"):
            fixture.build(db_path=tmp_path / "x.db", sql_path=tmp_path / "missing.sql")
