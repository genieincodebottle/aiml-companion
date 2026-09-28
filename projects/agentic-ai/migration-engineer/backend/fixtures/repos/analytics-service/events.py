"""Event timestamping for the analytics-service (deprecated utcnow, no timezone import)."""

from datetime import datetime


def event_time() -> datetime:
    return datetime.utcnow()
