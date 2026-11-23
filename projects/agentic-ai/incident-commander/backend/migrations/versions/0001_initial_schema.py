"""Initial schema: users, incidents, incident events, runbook episodes.

Revision ID: 0001
Revises: (none)
Create Date: 2026-11-23

Generated with `alembic revision --autogenerate` from src/db/models.py. Databases
created before migrations existed (by `create_all`) are adopted at startup by
stamping this revision, not by re-running it; see src/db/session.py.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "runbook_episodes",
        sa.Column("episode_id", sa.String(length=64), nullable=False),
        sa.Column("incident_id", sa.String(length=64), nullable=True),
        sa.Column("scenario_id", sa.String(length=64), nullable=False),
        sa.Column("service", sa.String(length=128), nullable=False),
        sa.Column("root_cause", sa.Text(), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("lesson", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("episode_id"),
    )
    op.create_index(op.f("ix_runbook_episodes_incident_id"), "runbook_episodes", ["incident_id"], unique=False)
    op.create_index(op.f("ix_runbook_episodes_outcome"), "runbook_episodes", ["outcome"], unique=False)
    op.create_index(op.f("ix_runbook_episodes_recorded_at"), "runbook_episodes", ["recorded_at"], unique=False)
    op.create_index(op.f("ix_runbook_episodes_scenario_id"), "runbook_episodes", ["scenario_id"], unique=False)
    op.create_index(op.f("ix_runbook_episodes_schema_version"), "runbook_episodes", ["schema_version"], unique=False)
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)
    op.create_table(
        "incidents",
        sa.Column("incident_id", sa.String(length=64), nullable=False),
        sa.Column("scenario_id", sa.String(length=64), nullable=False),
        sa.Column("service", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=True),
        sa.Column("halt_reason", sa.String(length=48), nullable=True),
        sa.Column("engine", sa.String(length=16), nullable=True),
        sa.Column("iterations", sa.Integer(), nullable=False),
        sa.Column("evidence_rounds", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("total_cost_usd", sa.Float(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("incident_id"),
    )
    op.create_index(op.f("ix_incidents_halt_reason"), "incidents", ["halt_reason"], unique=False)
    op.create_index(op.f("ix_incidents_scenario_id"), "incidents", ["scenario_id"], unique=False)
    op.create_index(op.f("ix_incidents_started_at"), "incidents", ["started_at"], unique=False)
    op.create_index(op.f("ix_incidents_status"), "incidents", ["status"], unique=False)
    op.create_table(
        "incident_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("incident_id", sa.String(length=64), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("agent", sa.String(length=32), nullable=True),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.incident_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("incident_id", "seq", name="uq_incident_event_seq"),
    )
    op.create_index(op.f("ix_incident_events_incident_id"), "incident_events", ["incident_id"], unique=False)
    op.create_index("ix_incident_events_incident_seq", "incident_events", ["incident_id", "seq"], unique=False)
    op.create_index(op.f("ix_incident_events_type"), "incident_events", ["type"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_incident_events_type"), table_name="incident_events")
    op.drop_index("ix_incident_events_incident_seq", table_name="incident_events")
    op.drop_index(op.f("ix_incident_events_incident_id"), table_name="incident_events")
    op.drop_table("incident_events")
    op.drop_index(op.f("ix_incidents_status"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_started_at"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_scenario_id"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_halt_reason"), table_name="incidents")
    op.drop_table("incidents")
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_table("users")
    op.drop_index(op.f("ix_runbook_episodes_schema_version"), table_name="runbook_episodes")
    op.drop_index(op.f("ix_runbook_episodes_scenario_id"), table_name="runbook_episodes")
    op.drop_index(op.f("ix_runbook_episodes_recorded_at"), table_name="runbook_episodes")
    op.drop_index(op.f("ix_runbook_episodes_outcome"), table_name="runbook_episodes")
    op.drop_index(op.f("ix_runbook_episodes_incident_id"), table_name="runbook_episodes")
    op.drop_table("runbook_episodes")
