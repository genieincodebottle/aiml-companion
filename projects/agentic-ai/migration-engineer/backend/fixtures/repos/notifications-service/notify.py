"""Notification timing for the notifications-service (deprecated utcnow, no timezone import)."""

from datetime import datetime


def queued_at() -> datetime:
    return datetime.utcnow()


def queued_at_iso() -> str:
    return datetime.utcnow().isoformat()
