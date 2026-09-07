"""Box, track, and alert types, with the coordinate convention stated once."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PPEClass(str, Enum):
    PERSON = "person"
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    VEST = "vest"
    NO_VEST = "no_vest"


#: The classes that constitute a violation when found inside a zone.
VIOLATION_CLASSES = (PPEClass.NO_HELMET, PPEClass.NO_VEST)


class Box(BaseModel):
    """xyxy in ORIGINAL image pixels, never letterboxed coordinates.

    Mixing the two is the most common bug in a detection pipeline, so
    the convention is in the type and undone as early as possible.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    cls: PPEClass
    conf: float = Field(ge=0.0, le=1.0)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom centre, where the person is standing. See src/zones.py."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def area(self) -> float:
        return max(self.x2 - self.x1, 0.0) * max(self.y2 - self.y1, 0.0)


class Track(BaseModel):
    track_id: int
    box: Box
    age_frames: int = 0
    lost_frames: int = 0
    first_seen_ts: float = 0.0


class Alert(BaseModel):
    track_id: int
    camera_id: str
    violation: PPEClass
    zone: str
    dwell_s: float
    clip_path: str
    start_ts: float = 0.0
    ts: float = 0.0


class Event(BaseModel):
    """Ground truth. A real violation that happened on the site."""

    zone: str
    violation: PPEClass
    start_ts: float
    end_ts: float
    track_id: int = -1
