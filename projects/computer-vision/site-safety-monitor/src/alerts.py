"""Assembles the alert, writes the clip, and deduplicates against recent
alerts."""
from __future__ import annotations

from src.config import settings
from src.schemas import Alert


class AlertSink:
    def __init__(self, buffer, store, cooldown_s: float | None = None):
        self.buffer = buffer
        self.store = store
        self.cooldown = (settings.alert_cooldown_s if cooldown_s is None
                         else cooldown_s)
        self.recent: dict[tuple[str, str], float] = {}
        self.fired: list[Alert] = []
        self.suppressed = 0

    def fire(self, track, zone, violation, camera_id, ts) -> Alert | None:
        # A second person entering the same zone within the cooldown is
        # still worth reporting, so the key includes the violation type
        # but NOT the track. Same zone, same violation, recently seen.
        key = (zone, getattr(violation, "value", violation))
        if ts - self.recent.get(key, -1e18) < self.cooldown:
            self.suppressed += 1
            return None
        self.recent[key] = ts

        path = self.store.write_clip(self.buffer.snapshot(), track.track_id)
        alert = Alert(track_id=track.track_id, camera_id=camera_id,
                      violation=violation, zone=zone,
                      dwell_s=ts - track.first_seen_ts, clip_path=path,
                      start_ts=track.first_seen_ts, ts=ts)
        self.fired.append(alert)
        return alert
