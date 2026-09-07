"""Decides which frames get a detection and which get tracked only."""
from __future__ import annotations

from src.config import settings


def should_detect(frame_index: int, has_active_tracks: bool,
                  detect_every: int | None = None) -> bool:
    """Every Nth frame, but always detect when nothing is being tracked.

    Skipping while no tracks exist means a person entering the scene
    waits up to N frames to be seen at all, and at the boundary of a
    restricted zone that is the moment that matters most.
    """
    if not has_active_tracks:
        return True
    n = settings.detect_every if detect_every is None else detect_every
    return frame_index % n == 0
