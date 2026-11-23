"""Relational schema for users, incidents and episodic memory.

Postgres is **optional**. With no `IC_DATABASE_URL` the app runs exactly as it
always has: incidents live in memory, runbook memory is a JSON file, and auth is
off. That is deliberate and worth preserving - a teaching project that cannot be
started without provisioning a database has raised the cost of the first run for
no pedagogical gain.

What Postgres buys when it *is* configured:
  * incidents survive a restart, so the console can show yesterday's runs
  * runbook memory becomes shared across processes rather than per-file
  * there is somewhere to put users, which is what makes login possible

Design notes:
  * `IncidentRow.state` keeps the full `IncidentState` as JSONB. The relational
    columns beside it are a projection for querying (list, filter, aggregate).
    Duplicating rather than normalising is intentional: the agent contract is
    still Pydantic, and shredding it into 20 tables would mean a migration every
    time an agent gains a field.
  * Events are a separate table because they are append-only and read by range
    (SSE resume asks for "everything after seq N"), which a JSONB array handles
    badly.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    """An operator who can log in and approve actions.

    `role` is the *application* role (viewer / operator / admin), which is not the
    same thing as `IC_AGENT_ROLE` - that one governs what the AGENT may execute.
    Keeping them separate matters: a human admin does not widen the agent's
    permissions, and an agent restricted to read-only stays read-only no matter who
    is logged in.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    #: bcrypt hash. The plaintext never leaves the request handler.
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="operator")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_public(self) -> dict:
        """Everything safe to hand to a browser. Note what is absent."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class IncidentRow(Base):
    """One incident run, with the full state document alongside query columns."""

    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String(64), index=True)
    service: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    halt_reason: Mapped[str | None] = mapped_column(String(48), nullable=True, index=True)
    engine: Mapped[str | None] = mapped_column(String(16), nullable=True)

    iterations: Mapped[int] = mapped_column(Integer, default=0)
    evidence_rounds: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Who triggered it. Null when auth is off, which is the default.
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    #: The complete IncidentState, exactly as the API serves it.
    state: Mapped[dict] = mapped_column(JSONB, default=dict)

    events: Mapped[list["IncidentEventRow"]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", lazy="selectin"
    )


class IncidentEventRow(Base):
    """One streamed loop event. Append-only; read by `seq` range for SSE resume."""

    __tablename__ = "incident_events"
    __table_args__ = (
        UniqueConstraint("incident_id", "seq", name="uq_incident_event_seq"),
        Index("ix_incident_events_incident_seq", "incident_id", "seq"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.incident_id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(32), index=True)
    agent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    incident: Mapped[IncidentRow] = relationship(back_populates="events")


class EpisodeRow(Base):
    """A runbook lesson, mirroring `memory.runbook.Episode`.

    Stored relationally as well as in the JSON file so that retrieval works across
    processes. `schema_version` is filtered on read for the same reason the file
    store filters it: an episode written under an older agent contract may describe
    actions that no longer exist.
    """

    __tablename__ = "runbook_episodes"

    episode_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    incident_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    scenario_id: Mapped[str] = mapped_column(String(64), index=True)
    service: Mapped[str] = mapped_column(String(128))
    root_cause: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(String(32), default="", index=True)
    lesson: Mapped[str] = mapped_column(Text, default="")
    schema_version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    #: The whole Episode dict, so retrieval can rebuild it without column drift.
    data: Mapped[dict] = mapped_column(JSONB, default=dict)
