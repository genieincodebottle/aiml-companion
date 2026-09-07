"""ByteTrack association, including the low-confidence second pass that gives it
its name."""
from __future__ import annotations

import numpy as np

from src.config import settings
from src.schemas import Box, Track

#: The second-pass floor. Boxes below this are noise, not occlusion.
LOW_CONF = 0.10

# There is NO module-level high threshold. It derives from
# settings.conf_threshold, because two independent confidence gates in one
# pipeline means the stricter one silently wins.
#
# This was a real bug here. The tracker held its own 0.50 while the config
# said 0.45, so only boxes above 0.50 could ever start a track and the
# whole confidence sweep came out flat from 0.15 to 0.45. The knob in the
# config file did nothing and nothing errored.


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class ByteTracker:
    """Two-stage association.

    High-confidence boxes match existing tracks first. The leftovers are
    then matched against LOW-confidence boxes, which is what recovers a
    partially occluded worker instead of ending the track and starting
    a new one with a new ID.
    """

    def __init__(self, iou_match: float = 0.30, high_conf: float | None = None):
        self.tracks: list[Track] = []
        self.iou_match = iou_match
        self.high_conf = (settings.conf_threshold if high_conf is None
                          else high_conf)
        self._next_id = 1

    def has_tracks(self) -> bool:
        return bool(self.tracks)

    def _associate(self, tracks, boxes, ts):
        """Greedy IoU matching, highest overlap first."""
        if not tracks or not boxes:
            return [], list(tracks), list(boxes)

        scores = np.array([[_iou(t.box, b) for b in boxes] for t in tracks])
        matched, used_t, used_b = [], set(), set()
        while True:
            i, j = np.unravel_index(np.argmax(scores), scores.shape)
            if scores[i, j] < self.iou_match:
                break
            t, b = tracks[i], boxes[j]
            t.box = b
            t.age_frames += 1
            t.lost_frames = 0
            matched.append(t)
            used_t.add(i)
            used_b.add(j)
            scores[i, :] = -1.0
            scores[:, j] = -1.0
            if len(used_t) == len(tracks) or len(used_b) == len(boxes):
                break
        return (matched,
                [t for i, t in enumerate(tracks) if i not in used_t],
                [b for j, b in enumerate(boxes) if j not in used_b])

    def update(self, boxes, frame_idx: int, ts: float) -> list[Track]:
        high = [b for b in boxes if b.conf >= self.high_conf]
        low = [b for b in boxes if LOW_CONF <= b.conf < self.high_conf]

        matched, unmatched_tracks, leftover_high = self._associate(
            self.tracks, high, ts)
        recovered, still_unmatched, _ = self._associate(
            unmatched_tracks, low, ts)

        # A frame with no detections is not a frame with no tracks. Every
        # unmatched track ages rather than ending, which is what carries
        # identity through the skipped frames.
        #
        # This tracker HOLDS the last box rather than predicting the next
        # one. A Kalman step would place the track where the person is
        # about to be, which matters on a skipped frame at 15 fps and
        # matters more the faster the subject moves. Holding is enough at
        # this frame rate and this walking speed, and saying so is better
        # than implying a motion model that is not here.
        for t in still_unmatched:
            t.lost_frames += 1

        started = []
        for b in leftover_high:
            started.append(Track(track_id=self._next_id, box=b,
                                 age_frames=1, first_seen_ts=ts))
            self._next_id += 1

        self.tracks = [t for t in matched + recovered + still_unmatched + started
                       if t.lost_frames <= settings.track_lost_max]
        return [t for t in self.tracks if t.lost_frames == 0]
