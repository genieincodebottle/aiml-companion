"""Holds a class label across frames by vote, so one bad frame does not flip a
violation."""
from __future__ import annotations

from collections import Counter, deque

from src.config import settings


class ClassVote:
    """A track's PPE state is the majority over its last N observations.

    A single frame where a helmet is occluded should not fire a no_helmet
    alert. Fifteen frames at 15 fps is one second of evidence.
    """

    def __init__(self, window: int | None = None):
        self.history: dict[int, deque] = {}
        self.window = window or settings.class_vote_window

    def update(self, track_id: int, cls) -> str:
        h = self.history.setdefault(track_id, deque(maxlen=self.window))
        h.append(getattr(cls, "value", cls))
        return Counter(h).most_common(1)[0][0]

    def forget(self, track_id: int) -> None:
        """Called when a track ends, or the dict grows for the whole shift."""
        self.history.pop(track_id, None)
