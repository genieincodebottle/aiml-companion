"""Alembic environment for the optional Postgres schema.

Three ways in:

* **The app, at startup.** `src/db/session.py` opens a connection and hands it over
  in `config.attributes["connection"]`; migrations run on it. This is the normal path.
* **The CLI, online.** `uv run alembic upgrade head` connects using IC_DATABASE_URL.
* **The CLI, offline.** `uv run alembic upgrade head --sql` prints the SQL without a
  database, which is also how the test suite checks the migrations.

The target metadata is `src.db.models.Base`, so `alembic revision --autogenerate`
compares the database with the models.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.engine import Connection

from src.config import get_settings
from src.db.models import Base
from src.db.session import _normalize

config = context.config
target_metadata = Base.metadata

#: Used only to pick the SQL dialect when printing SQL with no database configured.
_OFFLINE_PLACEHOLDER_URL = "postgresql://user@localhost/incident_commander"


def _database_url() -> str:
    url = get_settings().database_url
    return _normalize(url) if url else ""


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`--sql`)."""
    context.configure(
        url=_database_url() or _OFFLINE_PLACEHOLDER_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_with(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    url = _database_url()
    if not url:
        raise SystemExit("IC_DATABASE_URL is not set; nothing to migrate (use --sql to print the SQL instead).")
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        await conn.run_sync(_run_with)
        await conn.commit()
    await engine.dispose()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_with(connection)  # handed over by the app at startup
    else:
        asyncio.run(_run_async())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
