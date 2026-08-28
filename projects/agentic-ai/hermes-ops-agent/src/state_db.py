"""Read-only access to a Hermes Agent state database.

Hermes keeps everything it has ever done in a single SQLite file, by default
``~/.hermes/state.db``. The tables this module touches:

    sessions             one row per conversation, with token counts,
                         tool_call_count, api_call_count and cost columns
    messages             one row per turn, with role, tool_calls, tool_name
    session_model_usage  per-model attribution (not read here)

Two rules this module holds:

1.  **It opens the database read-only.** That is the whole reason for the
    ``mode=ro`` URI below. The file being read is the learner's live agent
    memory, and an analysis tool has no business writing to it. A stray write
    here would corrupt real sessions, and SQLite will happily let you do it.

2.  **It tolerates a schema it does not fully know.** Hermes ships fast and
    the sessions table has grown columns across migrations. Every read goes
    through ``_columns()`` so a missing column yields ``None`` instead of an
    OperationalError on someone's older or newer install.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote


DEFAULT_DB_RELATIVE = Path(".hermes") / "state.db"


def default_db_path() -> Path:
    """Where Hermes keeps state.db, honouring HERMES_HOME if it is set.

    Hermes resolves its home through hermes_constants.get_hermes_home(), which
    respects HERMES_HOME. Profiles (`hermes --profile work`) get their own home,
    so pointing this at a profile directory analyses that profile alone.
    """
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser() / "state.db"
    return Path.home() / DEFAULT_DB_RELATIVE


@dataclass(frozen=True)
class Session:
    """One Hermes conversation, reduced to the fields this project measures."""

    id: str
    source: str | None
    model: str | None
    title: str | None
    started_at: float | None
    ended_at: float | None
    message_count: int = 0
    tool_call_count: int = 0
    api_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = None
    parent_session_id: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return max(self.ended_at - self.started_at, 0.0)


# Columns we ask for, in Session field order. Anything absent from the live
# schema is filled with the dataclass default rather than raising.
_SESSION_COLUMNS = (
    "id",
    "source",
    "model",
    "title",
    "started_at",
    "ended_at",
    "message_count",
    "tool_call_count",
    "api_call_count",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "estimated_cost_usd",
    "parent_session_id",
)

_INT_FIELDS = {
    "message_count",
    "tool_call_count",
    "api_call_count",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
}


class StateDbError(RuntimeError):
    """Raised when the database is missing or is not a Hermes state store."""


@dataclass
class StateDb:
    """A read-only handle on one state.db.

    Use as a context manager::

        with StateDb.open() as db:
            for s in db.sessions(limit=5):
                print(s.id, s.tool_call_count)
    """

    path: Path
    _conn: sqlite3.Connection | None = field(default=None, repr=False)

    @classmethod
    def open(cls, path: Path | str | None = None) -> "StateDb":
        p = Path(path).expanduser() if path else default_db_path()
        if not p.exists():
            raise StateDbError(
                f"No Hermes state database at {p}.\n"
                "Run a Hermes session first, or point --db at a fixture:\n"
                "  make demo   (uses the recorded fixture, no Hermes needed)"
            )
        # mode=ro is load-bearing. See the module docstring.
        #
        # The path has to be percent-encoded first. In a SQLite URI, `?` opens
        # the query string and `#` opens a fragment, so a directory called
        # "notes#2" silently truncates the path and SQLite opens something
        # else entirely - which surfaced as a baffling "not a Hermes state
        # store" error rather than anything about the path. `/` and `:` stay
        # literal so Windows drive letters keep working.
        uri = f"file:{quote(p.as_posix(), safe='/:')}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            # Force a real read now. sqlite3.connect is lazy, so a directory
            # or a text file connects fine and only fails on first use -
            # which would surface as a traceback from somewhere unrelated.
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        except sqlite3.Error as e:
            raise StateDbError(
                f"Could not read {p} as a SQLite database ({e}).\n"
                "Point --db at a state.db file, not a directory, and check "
                "the file is not empty or truncated."
            ) from e
        db = cls(path=p, _conn=conn)
        db._assert_is_hermes_db()
        return db

    def __enter__(self) -> "StateDb":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StateDbError("StateDb is closed")
        return self._conn

    def _assert_is_hermes_db(self) -> None:
        names = {
            r["name"]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not names:
            # A zero-byte file is a *valid* empty SQLite database, so it gets
            # this far. Saying "not a Hermes state store" for it is technically
            # true and completely unhelpful.
            self.close()
            raise StateDbError(
                f"{self.path} is an empty database with no tables. "
                "If this is your real state.db, Hermes has not written to it yet."
            )
        missing = {"sessions", "messages"} - names
        if missing:
            self.close()
            raise StateDbError(
                f"{self.path} is a SQLite file but not a Hermes state store "
                f"(missing table(s): {', '.join(sorted(missing))})."
            )

    def _columns(self, table: str) -> set[str]:
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def schema_version(self) -> int | None:
        """Hermes tracks migrations in a single-row schema_version table."""
        try:
            row = self.conn.execute("SELECT * FROM schema_version LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        for key in row.keys():
            if "version" in key.lower():
                try:
                    return int(row[key])
                except (TypeError, ValueError):
                    return None
        return None

    def sessions(
        self,
        *,
        source: str | None = None,
        limit: int | None = None,
        newest_first: bool = True,
    ) -> list[Session]:
        available = self._columns("sessions")
        cols = [c for c in _SESSION_COLUMNS if c in available]
        if "id" not in cols:
            raise StateDbError("sessions table has no id column")

        sql = f"SELECT {', '.join(cols)} FROM sessions"
        params: list[object] = []
        if source is not None:
            if "source" not in available:
                return []
            sql += " WHERE source = ?"
            params.append(source)
        if "started_at" in available:
            sql += f" ORDER BY started_at {'DESC' if newest_first else 'ASC'}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        out = []
        for row in self.conn.execute(sql, params):
            out.append(self._to_session(row, cols))
        return out

    def session(self, session_id: str) -> Session | None:
        available = self._columns("sessions")
        cols = [c for c in _SESSION_COLUMNS if c in available]
        row = self.conn.execute(
            f"SELECT {', '.join(cols)} FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._to_session(row, cols) if row else None

    @staticmethod
    def _to_session(row: sqlite3.Row, cols: list[str]) -> Session:
        kwargs: dict[str, object] = {}
        for c in cols:
            v = row[c]
            if c in _INT_FIELDS:
                kwargs[c] = int(v) if v is not None else 0
            elif c == "estimated_cost_usd":
                kwargs[c] = float(v) if v is not None else None
            elif c in ("started_at", "ended_at"):
                kwargs[c] = float(v) if v is not None else None
            else:
                kwargs[c] = v
        kwargs.setdefault("source", None)
        kwargs.setdefault("model", None)
        kwargs.setdefault("title", None)
        kwargs.setdefault("started_at", None)
        kwargs.setdefault("ended_at", None)
        return Session(**kwargs)  # type: ignore[arg-type]

    def counted_tool_calls(self, session_id: str) -> int | None:
        """Count tool calls from the messages table, or None if we cannot.

        Why this exists when sessions.tool_call_count is right there: that
        column is a running counter Hermes maintains, and it is the number to
        trust. This recount from messages is the cross-check. If the two
        disagree on a session, something ended the session abnormally, and the
        report says so rather than quietly picking one.

        Returning **None** rather than 0 when the schema has no `tool_name`
        column is the important part. Zero is a real recount meaning "no tools
        ran"; a schema that cannot answer the question is not the same thing.
        Conflating them made every session on an older install report a
        mismatch against its own counter and print a spurious "ended
        abnormally" warning. None tells the caller to skip the cross-check.
        """
        available = self._columns("messages")
        if "tool_name" not in available or "role" not in available:
            return None
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE session_id = ? AND role = 'tool'",
            (session_id,),
        ).fetchone()
        return int(row["n"])

    def tool_names(self, session_id: str) -> dict[str, int]:
        """Which tools ran, and how often. This is what a skill changes."""
        available = self._columns("messages")
        if "tool_name" not in available:
            return {}
        rows = self.conn.execute(
            "SELECT tool_name, COUNT(*) AS n FROM messages "
            "WHERE session_id = ? AND tool_name IS NOT NULL "
            "GROUP BY tool_name ORDER BY n DESC",
            (session_id,),
        )
        return {r["tool_name"]: int(r["n"]) for r in rows}
