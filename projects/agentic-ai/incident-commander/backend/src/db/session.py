"""Async engine / session management, and the "is a database configured?" seam.

Everything here is written so that **no database is a supported configuration**,
not an error path. `db_enabled()` is false when `IC_DATABASE_URL` is empty, and
every caller is expected to check it rather than catch an exception - a project
whose default run requires Docker has made its first five minutes worse for
everyone learning from it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..config import BACKEND_DIR, get_settings

_ALEMBIC_INI = BACKEND_DIR / "alembic.ini"
#: The first migration, which matches the schema `create_all` used to build.
_BASELINE_REVISION = "0001"

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None
_ready = False


def db_enabled() -> bool:
    """True when a database URL is configured. Cheap; safe to call per request."""
    return bool(get_settings().database_url)


def _normalize(url: str) -> str:
    """Accept the URL forms people actually paste.

    `postgres://` and `postgresql://` are what every hosting dashboard hands you,
    but SQLAlchemy's async engine needs an async driver in the scheme. Rewriting it
    here means a copied connection string works rather than failing with a driver
    error that reads like a bug in this app.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def get_engine() -> AsyncEngine | None:
    global _engine, _sessionmaker
    if not db_enabled():
        return None
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            _normalize(settings.database_url),
            echo=False,
            pool_pre_ping=True,  # a Postgres restart should not poison the pool
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _engine


def _migrate(connection) -> str:
    """Bring the schema to the latest Alembic migration. Returns the revision.

    Runs on the app's own connection, so there is no second database login and no
    separate deploy step: starting the backend IS the migration. A database that
    predates migrations (tables made by `create_all`, no `alembic_version`) is
    adopted by stamping the initial revision, never by re-creating its tables.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import inspect

    cfg = Config(str(_ALEMBIC_INI))
    cfg.attributes["connection"] = connection
    tables = set(inspect(connection).get_table_names())
    if "incidents" in tables and "alembic_version" not in tables:
        logger.info("adopting a pre-migration database: stamping revision %s", _BASELINE_REVISION)
        command.stamp(cfg, _BASELINE_REVISION)
    command.upgrade(cfg, "head")
    return MigrationContext.configure(connection).get_current_revision() or "none"


async def init_db() -> bool:
    """Migrate the schema if a database is configured. Returns True when usable.

    Failure is logged and swallowed: if Postgres is down, the app degrades to
    in-memory rather than refusing to start. That is the opposite of the choice
    made for the Gemini key (ADR-0009) and for a reason - without a model this
    system is not agentic and has nothing to demonstrate, whereas without a
    database it is merely forgetful.
    """
    global _ready
    engine = get_engine()
    if engine is None:
        return False
    try:
        async with engine.begin() as conn:
            revision = await conn.run_sync(_migrate)
        _ready = True
        logger.info("database ready: schema at migration %s", revision)
    except Exception as exc:  # noqa: BLE001 - degradation is the designed behaviour
        _ready = False
        logger.warning("database unavailable, continuing in-memory: %s", exc)
    return _ready


def db_ready() -> bool:
    """True when `init_db` succeeded. False means "configured but unreachable"."""
    return _ready


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional session. Yields nothing useful if the DB is not ready."""
    if _sessionmaker is None or not _ready:
        raise RuntimeError("database is not configured or not ready")
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_db() -> None:
    global _engine, _sessionmaker, _ready
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker, _ready = None, None, False


def reset_db_state_for_tests() -> None:
    """Test hook: forget the cached engine so a new URL takes effect."""
    global _engine, _sessionmaker, _ready
    _engine, _sessionmaker, _ready = None, None, False
