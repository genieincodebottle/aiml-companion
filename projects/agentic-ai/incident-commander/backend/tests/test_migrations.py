"""The Alembic migrations build exactly the schema the models describe.

Runs with no database: Alembic's offline mode (`upgrade head --sql`) renders the
migration as Postgres SQL, and these tests read that SQL. A real-database check
(`alembic check`, "no new upgrade operations") is in docs/persistence-and-login.md.
"""

from __future__ import annotations

import contextlib
import io

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from src.config import BACKEND_DIR
from src.db.models import Base


def _config() -> Config:
    return Config(str(BACKEND_DIR / "alembic.ini"))


def _offline_sql() -> str:
    buffer = io.StringIO()
    cfg = _config()
    cfg.output_buffer = buffer
    with contextlib.redirect_stdout(buffer):
        command.upgrade(cfg, "head", sql=True)
    return buffer.getvalue()


def test_there_is_exactly_one_migration_head():
    """Two heads means two branches of schema history that nobody merged."""
    assert len(ScriptDirectory.from_config(_config()).get_heads()) == 1


def test_the_migrations_create_every_table_the_models_define():
    sql = _offline_sql()
    for table in Base.metadata.tables:
        assert f"CREATE TABLE {table}" in sql, f"no migration creates table '{table}'"


def test_the_migrations_create_every_index_the_models_define():
    sql = _offline_sql()
    for table in Base.metadata.tables.values():
        for index in table.indexes:
            assert index.name in sql, f"no migration creates index '{index.name}'"
