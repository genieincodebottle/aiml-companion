"""A rolling buffer so the seconds before an alert can be saved, not just the
seconds after."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path


class ClipBuffer:
    """Keeps the last N seconds in memory at all times.

    The useful part of a safety clip is what happened BEFORE the alert.
    Starting to record when the alert fires captures the aftermath and
    loses the approach, which is the part that shows intent.
    """

    def __init__(self, fps: int, seconds: int = 7):
        self.frames = deque(maxlen=fps * seconds)
        self.seconds = seconds

    def push(self, frame, ts):
        self.frames.append((frame, ts))

    def snapshot(self):
        return list(self.frames)


class ClipStore:
    """Writes a manifest rather than an encoded video.

    The demo has no encoder and no real frames. What the alert needs is a
    path plus the time span it covers, and writing that honestly is
    better than shipping a stub that pretends to produce an MP4.
    """

    def __init__(self, root: str = "artifacts/clips"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_clip(self, frames, track_id: int) -> str:
        path = self.root / f"track_{track_id}.json"
        span = ([frames[0][1], frames[-1][1]] if frames else [0.0, 0.0])
        path.write_text(json.dumps({
            "track_id": track_id,
            "frames": len(frames),
            "from_ts": span[0],
            "to_ts": span[1],
        }))
        return str(path).replace("\\", "/")
