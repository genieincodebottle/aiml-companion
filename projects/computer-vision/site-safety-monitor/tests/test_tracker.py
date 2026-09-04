"""Identity across frames, and the two thresholds that must not fight."""
from __future__ import annotations

from src.config import settings
from src.schemas import PPEClass
from src.track_smoothing import ClassVote
from src.tracker import LOW_CONF, ByteTracker
from tests.conftest import box


def test_a_walking_worker_keeps_one_id():
    t = ByteTracker(high_conf=0.45)
    ids = set()
    for i in range(30):
        tracks = t.update([box(300.0 + i * 1.5, 430.0)], i, i / 15)
        ids.update(x.track_id for x in tracks)
    assert ids == {1}


def test_an_empty_frame_does_not_end_the_track():
    """The skipped frames pass an empty list, and identity has to survive it."""
    t = ByteTracker(high_conf=0.45)
    t.update([box(300.0, 430.0)], 0, 0.0)
    for i in range(1, 10):
        t.update([], i, i / 15)                 # a skipped frame
    assert t.has_tracks()
    tracks = t.update([box(300.0, 430.0)], 10, 10 / 15)
    assert tracks[0].track_id == 1              # same person, same id


def test_a_low_confidence_box_recovers_a_track_rather_than_ending_it():
    """The second association pass is what ByteTrack is named for."""
    t = ByteTracker(high_conf=0.45)
    t.update([box(300.0, 430.0, conf=0.9)], 0, 0.0)
    # Partially occluded: confidence drops but does not vanish.
    tracks = t.update([box(302.0, 430.0, conf=0.25)], 1, 1 / 15)
    assert tracks[0].track_id == 1


def test_noise_below_the_low_floor_is_not_used_at_all():
    t = ByteTracker(high_conf=0.45)
    t.update([box(300.0, 430.0, conf=0.9)], 0, 0.0)
    t.update([box(302.0, 430.0, conf=LOW_CONF / 2)], 1, 1 / 15)
    assert t.tracks[0].lost_frames == 1         # aged, not matched


def test_the_high_gate_follows_the_configured_threshold():
    """Two independent confidence gates means the stricter one silently wins.

    This is a regression test for a real bug. The tracker held its own
    0.50 while the config said 0.45, so the confidence sweep was flat
    from 0.15 to 0.45 and nothing errored.
    """
    assert ByteTracker().high_conf == settings.conf_threshold
    assert ByteTracker(high_conf=0.15).high_conf == 0.15


def test_a_box_below_the_gate_cannot_start_a_track():
    t = ByteTracker(high_conf=0.45)
    assert t.update([box(300.0, 430.0, conf=0.30)], 0, 0.0) == []
    t2 = ByteTracker(high_conf=0.15)
    assert len(t2.update([box(300.0, 430.0, conf=0.30)], 0, 0.0)) == 1


def test_one_bad_frame_does_not_flip_the_class():
    v = ClassVote(window=15)
    for _ in range(10):
        v.update(1, PPEClass.HELMET)
    assert v.update(1, PPEClass.NO_HELMET) == "helmet"   # outvoted


def test_a_sustained_change_does_flip_it():
    v = ClassVote(window=15)
    for _ in range(15):
        v.update(1, PPEClass.HELMET)
    for _ in range(8):
        out = v.update(1, PPEClass.NO_HELMET)
    assert out == "no_helmet"


class SpyTracker(ByteTracker):
    """Records how it was called, so the pipeline's contract can be asserted."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = []

    def update(self, boxes, frame_idx, ts):
        self.calls.append((frame_idx, len(boxes)))
        return super().update(boxes, frame_idx, ts)


def test_the_tracker_receives_every_frame_including_the_skipped_ones():
    """The design claim of the whole project, and it had no gate.

    A skipped frame passes an EMPTY box list to the tracker rather than
    skipping the tracker. Skipping it as well would break every track on
    every second frame, and nothing would error: the alerts would simply
    become worse in a way that reads as a weaker model.

    A canary that rewrote the pipeline to skip the tracker was caught by
    nothing in this suite until this test existed.
    """
    from src.pipeline import run
    from src.alerts import AlertSink
    from src.clip_buffer import ClipBuffer, ClipStore
    from src.dwell import DwellTimer
    from src.sim.scene import Frame, ZONES
    from src.zones import ZoneSet
    import tempfile

    frames = [Frame(idx=i, ts=i / 15, camera_id="cam_north",
                    boxes=[box(300.0 + i, 430.0)])
              for i in range(20)]
    spy = SpyTracker(high_conf=0.45)
    with tempfile.TemporaryDirectory() as d:
        sink = AlertSink(ClipBuffer(15, 7), ClipStore(d))
        run(frames, lambda f: f.boxes, spy, ClassVote(), ZoneSet(ZONES),
            DwellTimer(), sink, conf=0.45, detect_every=2)

    assert len(spy.calls) == len(frames), "the tracker missed a frame"
    empty = [c for c in spy.calls if c[1] == 0]
    assert empty, "no frame was skipped, so the claim was never exercised"
    # Frames 1, 3, ... 19: the odd half of twenty.
    assert len(empty) == 10
