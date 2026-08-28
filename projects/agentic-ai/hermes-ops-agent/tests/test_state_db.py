"""Tests for reading a Hermes state store.

The two behaviours worth defending here are that the reader never writes, and
that it survives a schema it does not recognise. Hermes ships fast; a column
appearing or disappearing must not take this tool down.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.state_db import StateDb, StateDbError, default_db_path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_SQL = ROOT / "fixtures" / "state.fixture.sql"


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> Path:
    db = tmp_path_factory.mktemp("state") / "state.fixture.db"
    conn = sqlite3.connect(db)
    conn.executescript(FIXTURE_SQL.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return db


class TestOpen:
    def test_reads_the_fixture(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            assert db.schema_version() == 22
            assert len(db.sessions()) == 3

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(StateDbError, match="No Hermes state database"):
            StateDb.open(tmp_path / "nope.db")

    def test_rejects_a_sqlite_file_that_is_not_hermes(self, tmp_path):
        other = tmp_path / "notes.db"
        conn = sqlite3.connect(other)
        conn.execute("CREATE TABLE notes (id INTEGER)")
        conn.commit()
        conn.close()
        with pytest.raises(StateDbError, match="not a Hermes state store"):
            StateDb.open(other)

    def test_default_path_follows_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profileA"))
        assert default_db_path() == tmp_path / "profileA" / "state.db"

    def test_default_path_without_hermes_home(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert default_db_path().name == "state.db"
        assert default_db_path().parent.name == ".hermes"


class TestReadOnly:
    def test_connection_refuses_writes(self, fixture_db):
        # This is the guarantee that matters: the file being read is somebody's
        # live agent memory. SQLite will let you corrupt it if you open it rw.
        with StateDb.open(fixture_db) as db:
            with pytest.raises(sqlite3.OperationalError):
                db.conn.execute("DELETE FROM sessions")

    def test_file_is_unchanged_after_a_full_read(self, fixture_db):
        before = fixture_db.read_bytes()
        with StateDb.open(fixture_db) as db:
            for s in db.sessions():
                db.tool_names(s.id)
                db.counted_tool_calls(s.id)
        assert fixture_db.read_bytes() == before


class TestSessions:
    def test_orders_newest_first_by_default(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            ids = [s.id for s in db.sessions()]
        assert ids[0] == "sess_gateway_0003"

    def test_filters_by_source(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            assert len(db.sessions(source="cli")) == 2
            assert len(db.sessions(source="telegram")) == 1
            assert db.sessions(source="signal") == []

    def test_limit(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            assert len(db.sessions(limit=1)) == 1

    def test_session_fields(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            s = db.session("sess_cold_0001")
        assert s.tool_call_count == 11
        assert s.cache_read_tokens == 0
        assert s.total_tokens == 51300
        assert s.duration_seconds == pytest.approx(742.0)

    def test_unknown_session_is_none(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            assert db.session("sess_nope") is None


class TestToolCounts:
    def test_recount_matches_the_stored_counter(self, fixture_db):
        # If these two disagree the report warns, so the fixture must not
        # accidentally disagree or every demo run shows a spurious warning.
        with StateDb.open(fixture_db) as db:
            for sid in ("sess_cold_0001", "sess_skill_0002"):
                s = db.session(sid)
                assert db.counted_tool_calls(sid) == s.tool_call_count

    def test_tool_names_breakdown(self, fixture_db):
        with StateDb.open(fixture_db) as db:
            cold = db.tool_names("sess_cold_0001")
            warm = db.tool_names("sess_skill_0002")
        assert cold["terminal"] == 5
        assert cold["web_search"] == 2
        assert "web_search" not in warm
        assert warm["skill_view"] == 1


class TestSchemaTolerance:
    def test_survives_a_sessions_table_missing_newer_columns(self, tmp_path):
        # An older install predating the cache-token columns must still work.
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                tool_call_count INTEGER DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
            INSERT INTO sessions VALUES ('s1', 'cli', 1.0, 3);
            """
        )
        conn.commit()
        conn.close()

        with StateDb.open(db) as sdb:
            s = sdb.session("s1")
            assert s.tool_call_count == 3
            assert s.cache_read_tokens == 0     # default, not a crash
            assert s.model is None
            assert sdb.schema_version() is None  # no schema_version table
            assert sdb.tool_names("s1") == {}    # no tool_name column


class TestCli:
    def test_demo_runs_and_reports_the_skill_win(self):
        r = subprocess.run(
            [sys.executable, "main.py", "demo"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert "skill shortened the procedure" in r.stdout
        assert "prompt caching" in r.stdout   # the confound is surfaced

    def test_sessions_listing(self):
        r = subprocess.run(
            [sys.executable, "main.py",
             "--db", "fixtures/state.fixture.db", "sessions"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert "sess_cold_0001" in r.stdout

    def test_missing_db_exits_two_with_a_useful_message(self, tmp_path):
        r = subprocess.run(
            [sys.executable, "main.py", "--db", str(tmp_path / "x.db"), "sessions"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 2
        assert "make demo" in r.stderr


class TestJsonOutput:
    def test_demo_json_is_machine_readable(self):
        # Regression: the fixture banner used to go to stdout and broke
        # `python main.py demo --json | jq`.
        import json

        r = subprocess.run(
            [sys.executable, "main.py", "demo", "--json"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)          # must not raise
        assert payload["verdict"] == "skill shortened the procedure"
        assert payload["baseline_session"]["tool_call_count"] == 11
        assert "# Recorded fixture" in r.stderr  # banner still shown, just not on stdout

    def test_compare_json_writes_a_report_file(self, tmp_path):
        import json

        out = tmp_path / "report.json"
        r = subprocess.run(
            [sys.executable, "main.py", "--db", "fixtures/state.fixture.db",
             "compare", "sess_cold_0001", "sess_skill_0002",
             "--json", "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        assert json.loads(out.read_text())["verdict"]


class TestRecountOnOlderSchemas:
    def test_missing_tool_name_column_returns_none_not_zero(self, tmp_path):
        # Zero means "no tools ran". None means "cannot answer". Conflating
        # them made every session on an older install report a mismatch
        # against its own counter and print a spurious warning.
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL,
                started_at REAL NOT NULL, tool_call_count INTEGER DEFAULT 0);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL, timestamp REAL NOT NULL);
            INSERT INTO sessions VALUES ('a', 'cli', 1.0, 11);
            """
        )
        conn.commit()
        conn.close()
        with StateDb.open(db) as sdb:
            assert sdb.counted_tool_calls("a") is None

    def test_no_spurious_abnormal_warning_on_an_older_schema(self, tmp_path):
        from src.metrics import compare

        db = tmp_path / "old2.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL,
                started_at REAL NOT NULL, tool_call_count INTEGER DEFAULT 0);
            CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL, timestamp REAL NOT NULL);
            INSERT INTO sessions VALUES ('a', 'cli', 1.0, 11);
            INSERT INTO sessions VALUES ('b', 'cli', 2.0, 4);
            """
        )
        conn.commit()
        conn.close()
        with StateDb.open(db) as sdb:
            c = compare(
                sdb.session("a"), sdb.session("b"),
                baseline_recount=sdb.counted_tool_calls("a"),
                candidate_recount=sdb.counted_tool_calls("b"),
            )
        assert not any("ended abnormally" in w for w in c.warnings), c.warnings

    def test_a_genuine_mismatch_still_warns(self, fixture_db):
        from src.metrics import compare

        with StateDb.open(fixture_db) as sdb:
            c = compare(
                sdb.session("sess_cold_0001"), sdb.session("sess_skill_0002"),
                baseline_recount=2,   # real recount, genuinely disagrees
                candidate_recount=4,
            )
        assert any("ended abnormally" in w for w in c.warnings)


class TestAwkwardInputs:
    """Every one of these used to escape as a raw sqlite3 traceback or a
    misleading message. main.py only catches StateDbError."""

    def test_path_containing_a_hash_still_opens(self, tmp_path):
        # `#` opens a fragment in a SQLite URI, so an unencoded path silently
        # truncated and SQLite opened something else.
        d = tmp_path / "notes#2"
        d.mkdir()
        target = d / "state.db"
        target.write_bytes((ROOT / "fixtures" / "state.fixture.db").read_bytes())
        with StateDb.open(target) as db:
            assert len(db.sessions()) == 3

    def test_path_containing_spaces_and_unicode(self, tmp_path):
        d = tmp_path / "my agent éç"
        d.mkdir()
        target = d / "state.db"
        target.write_bytes((ROOT / "fixtures" / "state.fixture.db").read_bytes())
        with StateDb.open(target) as db:
            assert len(db.sessions()) == 3

    def test_directory_instead_of_a_file(self, tmp_path):
        with pytest.raises(StateDbError, match="Could not read"):
            StateDb.open(tmp_path)

    def test_text_file_instead_of_a_database(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("not a database")
        with pytest.raises(StateDbError, match="Could not read"):
            StateDb.open(p)

    def test_zero_byte_file_says_it_is_empty(self, tmp_path):
        p = tmp_path / "empty.db"
        p.touch()
        with pytest.raises(StateDbError, match="empty database with no tables"):
            StateDb.open(p)


class TestSelfComparison:
    def test_comparing_a_session_with_itself_warns(self, fixture_db):
        from src.metrics import compare

        with StateDb.open(fixture_db) as db:
            s = db.session("sess_cold_0001")
            c = compare(s, s, baseline_recount=11, candidate_recount=11)
        assert any("same session" in w for w in c.warnings)
