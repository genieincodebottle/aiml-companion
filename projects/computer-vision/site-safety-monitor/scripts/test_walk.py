"""Weekly verification that the whole chain still works.

Someone walks each zone without a helmet, on purpose, and the script
checks that an alert arrived within the dwell time plus a margin. It is
the only test that covers the camera, the lens, the lighting, the model,
the zones, and the notification path at once.
"""
from __future__ import annotations

from src.config import settings

MARGIN_S = 5.0


def verify(zone: str, started_at: float, alerts) -> bool:
    expected_by = started_at + settings.dwell_seconds + MARGIN_S
    return any(a.zone == zone and a.ts <= expected_by for a in alerts)


def walk_report(zones, started_at, alerts):
    return {z: verify(z, started_at, alerts) for z in zones}
