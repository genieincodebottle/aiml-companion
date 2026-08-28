"""Build the demo fixture database from the checked-in SQL.

The `.sql` is the source of truth and is committed; the `.db` is a build
artefact and is gitignored, so that a reviewer can read what the demo claims
in a diff instead of opening SQLite.

The consequence is that a fresh clone has no `.db`. Nothing here should ever
make a learner deal with that. Building it takes about ten milliseconds and is
completely deterministic, so `main.py` and the test suite both call
`ensure()` and it just happens.

The alternative - telling the user to run `make fixture` first - fails on
Windows, where `make` is usually not installed, and turns the very first
command in the README into a dead end.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SQL_PATH = ROOT / "fixtures" / "state.fixture.sql"
DB_PATH = ROOT / "fixtures" / "state.fixture.db"


class FixtureError(RuntimeError):
    pass


def build(db_path: Path | None = None, sql_path: Path | None = None) -> Path:
    """(Re)build the fixture database, replacing any existing file."""
    sql = Path(sql_path) if sql_path else SQL_PATH
    db = Path(db_path) if db_path else DB_PATH

    if not sql.exists():
        raise FixtureError(
            f"Cannot build the demo fixture: {sql} is missing.\n"
            "It is checked into the repository, so this usually means the "
            "clone is incomplete."
        )

    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    conn = sqlite3.connect(db)
    try:
        conn.executescript(sql.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
    return db


def ensure(db_path: Path | None = None) -> Path:
    """Build the fixture only if it is not already there.

    Deliberately cheap to call on every run. A stale rebuild costs nothing and
    a missing fixture costs the learner their first command.
    """
    db = Path(db_path) if db_path else DB_PATH
    if db.exists() and db.stat().st_size > 0:
        return db
    return build(db)
