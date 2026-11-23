"""Persistence for incidents, events and runbook episodes.

Every function here is a no-op when no database is configured. Callers therefore
never branch on "do we have Postgres?" - they just call, and the seam absorbs it.
That keeps `loop.py` about the agent loop rather than about storage.

Failures are logged and swallowed for the same reason: losing the audit trail of a
run is bad, but killing a live incident because a write timed out is worse. The
in-memory copy remains the source of truth *during* a run; Postgres is what makes
it survive the process.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import EpisodeRow, IncidentEventRow, IncidentRow
from .session import db_ready, session_scope

logger = logging.getLogger(__name__)


def _parse_ts(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


async def save_incident(state_dict: dict, *, created_by: int | None = None) -> None:
    """Upsert an incident from its serialized `IncidentState`.

    Called on every status change rather than only at the end, so an incident that
    crashes mid-run still leaves a row explaining how far it got.
    """
    if not db_ready():
        return
    try:
        async with session_scope() as s:
            values = {
                "incident_id": state_dict["incident_id"],
                "scenario_id": state_dict.get("scenario_id", ""),
                "service": state_dict.get("service", ""),
                "status": state_dict.get("status", ""),
                "severity": state_dict.get("severity"),
                "halt_reason": state_dict.get("halt_reason"),
                "engine": state_dict.get("engine"),
                "iterations": state_dict.get("iterations", 0),
                "evidence_rounds": state_dict.get("evidence_rounds", 0),
                "total_tokens": state_dict.get("total_tokens", 0),
                "total_cost_usd": float(state_dict.get("total_cost_usd", 0.0) or 0.0),
                "started_at": _parse_ts(state_dict.get("started_at")) or dt.datetime.now(dt.timezone.utc),
                "ended_at": _parse_ts(state_dict.get("ended_at")),
                "state": state_dict,
            }
            if created_by is not None:
                values["created_by"] = created_by
            stmt = pg_insert(IncidentRow).values(**values)
            # The row is written many times per run, so the upsert updates every
            # mutable column rather than relying on the first write being complete.
            stmt = stmt.on_conflict_do_update(
                index_elements=[IncidentRow.incident_id],
                set_={k: v for k, v in values.items() if k != "incident_id"},
            )
            await s.execute(stmt)
    except Exception as exc:  # noqa: BLE001 - storage must never kill a live incident
        logger.warning("failed to persist incident %s: %s", state_dict.get("incident_id"), exc)


async def save_event(incident_id: str, event: dict) -> None:
    """Append one loop event. Duplicate (incident_id, seq) is ignored.

    Ignoring conflicts rather than erroring matters for the graph engine, which can
    replay a node after a checkpoint restore and legitimately re-emit a seq.
    """
    if not db_ready():
        return
    try:
        async with session_scope() as s:
            stmt = (
                pg_insert(IncidentEventRow)
                .values(
                    incident_id=incident_id,
                    seq=event.get("seq", 0),
                    type=event.get("type", ""),
                    agent=event.get("agent"),
                    iteration=event.get("iteration", 0),
                    payload=event.get("payload", {}),
                    ts=_parse_ts(event.get("ts")) or dt.datetime.now(dt.timezone.utc),
                )
                .on_conflict_do_nothing(index_elements=["incident_id", "seq"])
            )
            await s.execute(stmt)
    except Exception as exc:  # noqa: BLE001
        logger.debug("failed to persist event for %s: %s", incident_id, exc)


async def load_incident(incident_id: str) -> dict | None:
    if not db_ready():
        return None
    try:
        async with session_scope() as s:
            row = (
                await s.execute(select(IncidentRow).where(IncidentRow.incident_id == incident_id))
            ).scalar_one_or_none()
            return dict(row.state) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load incident %s: %s", incident_id, exc)
        return None


async def list_incidents(limit: int = 100) -> list[dict]:
    """Summaries, newest first. Built from the projection columns, not the JSONB."""
    if not db_ready():
        return []
    try:
        async with session_scope() as s:
            rows = (
                (await s.execute(select(IncidentRow).order_by(IncidentRow.started_at.desc()).limit(limit)))
                .scalars()
                .all()
            )
            return [
                {
                    "incident_id": r.incident_id,
                    "scenario_id": r.scenario_id,
                    "service": r.service,
                    "status": r.status,
                    "severity": r.severity,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                    "iterations": r.iterations,
                    "engine": r.engine,
                    "halt_reason": r.halt_reason,
                    "finalized": r.ended_at is not None,
                    "total_cost_usd": round(r.total_cost_usd or 0.0, 6),
                    "resolved": r.status == "resolved",
                    "persisted": True,
                }
                for r in rows
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to list incidents: %s", exc)
        return []


async def stats() -> dict:
    """Aggregate figures for the dashboard, computed in the database."""
    if not db_ready():
        return {}
    try:
        from sqlalchemy import func

        async with session_scope() as s:
            total = (await s.execute(select(func.count()).select_from(IncidentRow))).scalar_one()
            resolved = (
                await s.execute(select(func.count()).select_from(IncidentRow).where(IncidentRow.status == "resolved"))
            ).scalar_one()
            escalated = (
                await s.execute(select(func.count()).select_from(IncidentRow).where(IncidentRow.status == "escalated"))
            ).scalar_one()
            spend = (await s.execute(select(func.coalesce(func.sum(IncidentRow.total_cost_usd), 0.0)))).scalar_one()
            iters = (await s.execute(select(func.coalesce(func.avg(IncidentRow.iterations), 0.0)))).scalar_one()
            return {
                "total": int(total),
                "resolved": int(resolved),
                "escalated": int(escalated),
                "total_cost_usd": round(float(spend), 6),
                "mean_iterations": round(float(iters), 2),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to compute stats: %s", exc)
        return {}


# --- runbook memory ---------------------------------------------------------


async def save_episode(episode: dict) -> None:
    if not db_ready():
        return
    try:
        async with session_scope() as s:
            values = {
                "episode_id": episode.get("episode_id", ""),
                "incident_id": episode.get("incident_id"),
                "scenario_id": episode.get("scenario_id", ""),
                "service": episode.get("service", ""),
                "root_cause": episode.get("root_cause", "") or "",
                "outcome": episode.get("outcome", "") or "",
                "lesson": episode.get("lesson", "") or "",
                "schema_version": int(episode.get("schema_version", 1) or 1),
                "recorded_at": _parse_ts(episode.get("recorded_at")) or dt.datetime.now(dt.timezone.utc),
                "data": episode,
            }
            stmt = pg_insert(EpisodeRow).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[EpisodeRow.episode_id],
                set_={k: v for k, v in values.items() if k != "episode_id"},
            )
            await s.execute(stmt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist episode: %s", exc)


async def load_episodes(schema_version: int | None = None, limit: int = 500) -> list[dict]:
    if not db_ready():
        return []
    try:
        async with session_scope() as s:
            q = select(EpisodeRow).order_by(EpisodeRow.recorded_at.desc()).limit(limit)
            if schema_version is not None:
                q = q.where(EpisodeRow.schema_version == schema_version)
            rows = (await s.execute(q)).scalars().all()
            return [dict(r.data) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load episodes: %s", exc)
        return []


async def clear_episodes() -> None:
    """Used by the memory-reset endpoint and by tests."""
    if not db_ready():
        return
    async with session_scope() as s:
        await s.execute(delete(EpisodeRow))
