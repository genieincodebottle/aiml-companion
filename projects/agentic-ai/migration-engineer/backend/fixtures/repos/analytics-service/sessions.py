"""Session timestamping for the analytics-service (deprecated utcnow, no timezone import)."""

from datetime import datetime


def session_start() -> datetime:
    return datetime.utcnow()
