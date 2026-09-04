"""Requires a track to hold a violation for a configured time before it becomes
an alert."""
from __future__ import annotations

from src.config import settings


class DwellTimer:
    """Suppresses the flicker at a zone boundary.

    A worker crossing the edge of a zone for half a second is not a
    violation. Without this, standing near the line generates an alert
    every time the box wobbles across it.
    """

    def __init__(self, dwell_seconds: float | None = None):
        self.entered: dict[tuple[int, str], float] = {}
        self.alerted: set[tuple[int, str]] = set()
        self.dwell = (settings.dwell_seconds if dwell_seconds is None
                      else dwell_seconds)

    def should_alert(self, track_id, zone, ts) -> bool:
        key = (track_id, zone)
        if key in self.alerted:
            return False                       # one alert per track per zone
        first = self.entered.setdefault(key, ts)
        if ts - first >= self.dwell:
            self.alerted.add(key)
            return True
        return False

    def left(self, track_id, zone) -> None:
        """Clears the entry clock when a track leaves, so a worker who steps
        out and back in starts their dwell again rather than inheriting it."""
        self.entered.pop((track_id, zone), None)
