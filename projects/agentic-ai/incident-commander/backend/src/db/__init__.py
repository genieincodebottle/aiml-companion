"""Optional Postgres persistence and the user store that makes login possible.

Import this package freely: nothing here connects until `IC_DATABASE_URL` is set.
"""

from .session import db_enabled, db_ready, dispose_db, get_engine, init_db, session_scope

__all__ = ["db_enabled", "db_ready", "dispose_db", "get_engine", "init_db", "session_scope"]
