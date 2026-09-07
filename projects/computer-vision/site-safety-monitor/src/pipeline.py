"""The main loop, wired so the skipped frames genuinely skip the expensive
stages."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.budget import MODELLED_MS, modelled_frame_ms, stage
from src.config import settings
from src.frame_skip import should_detect
from src.schemas import VIOLATION_CLASSES, PPEClass

# Built once. Rebuilding this list inside the per-track loop was allocating on
# every frame in a pipeline whose whole subject is the frame budget.
VIOLATION_VALUES = frozenset(c.value for c in VIOLATION_CLASSES)


@dataclass
class RunStats:
    frames: int = 0
    detected_frames: int = 0
    skipped_frames: int = 0
    raw_violation_detections: int = 0
    alerts: int = 0
    suppressed_by_cooldown: int = 0
    suppressed_by_dwell: int = 0
    modelled_ms: float = 0.0
    zone_hits_foot: int = 0
    zone_hits_centre: int = 0
    fired: list = field(default_factory=list)

    @property
    def modelled_avg_ms(self) -> float:
        return self.modelled_ms / max(self.frames, 1)

    @property
    def reduction_ratio(self) -> float:
        return self.raw_violation_detections / max(self.alerts, 1)


def run(frames, detector_fn, tracker, votes, zones, dwell, sink,
        conf: float | None = None, detect_every: int | None = None,
        compare_centre_rule: bool = False) -> RunStats:
    """`detector_fn(frame) -> list[Box]` so the demo can feed detections
    directly and the device can feed the engine, with the same loop."""
    st = RunStats()
    conf = settings.conf_threshold if conf is None else conf

    # Names come from the ZoneSet rather than being written here. An earlier
    # version cleared the dwell clock for "crane_radius" and "excavation" by
    # name, so a third zone in configs/base.yaml would have kept a stale entry
    # clock forever and alerted early. Nothing would have errored.
    zone_names = tuple(zones.polys)

    for f in frames:
        detect = should_detect(f.idx, tracker.has_tracks(), detect_every)
        if detect:
            boxes = [b for b in detector_fn(f) if b.conf >= conf]
            st.detected_frames += 1
        else:
            boxes = []                             # tracker carries this frame
            st.skipped_frames += 1

        st.modelled_ms += modelled_frame_ms(detect)

        with stage("track"):
            tracks = tracker.update(boxes, f.idx, f.ts)

        with stage("zone"):
            for t in tracks:
                cls = votes.update(t.track_id, t.box.cls)
                zone = zones.zone_of(t.box)
                if zone:
                    st.zone_hits_foot += 1
                # The box-centre test is the WRONG rule, kept only so the demo
                # can quantify what it would have cost. It is a second polygon
                # test per track per frame, so it stays behind a flag rather
                # than running on the device inside a 33 ms budget.
                if compare_centre_rule and zones.zone_of_centre(t.box):
                    st.zone_hits_centre += 1
                if zone and cls in VIOLATION_VALUES:
                    st.raw_violation_detections += 1
                    if dwell.should_alert(t.track_id, zone, f.ts):
                        with stage("emit"):
                            a = sink.fire(t, zone, PPEClass(cls),
                                          f.camera_id, f.ts)
                        if a is not None:
                            st.alerts += 1
                    else:
                        st.suppressed_by_dwell += 1
                elif zone is None:
                    for z in zone_names:
                        dwell.left(t.track_id, z)

        st.frames += 1

    st.suppressed_by_cooldown = sink.suppressed
    st.fired = list(sink.fired)
    return st


def run_shift(frames_by_camera, detector_fn, zones_cfg, sink,
              conf: float | None = None, detect_every: int | None = None,
              dwell_seconds: float | None = None,
              compare_centre_rule: bool = False) -> RunStats:
    """One tracker per camera, one alert sink for the site.

    Tracking is per stream, because a track id only means anything within
    one view. Deduplication is site-wide, because two cameras seeing the
    same zone should not produce two alerts for one situation.
    """
    from src.dwell import DwellTimer
    from src.track_smoothing import ClassVote
    from src.tracker import ByteTracker
    from src.zones import ZoneSet

    total = RunStats()
    for _cam, frames in sorted(frames_by_camera.items()):
        # The tracker's high gate follows the configured threshold. See the
        # note at the top of src/tracker.py for why that is not optional.
        st = run(frames, detector_fn, ByteTracker(high_conf=conf), ClassVote(),
                 ZoneSet(zones_cfg), DwellTimer(dwell_seconds), sink,
                 conf=conf, detect_every=detect_every,
                 compare_centre_rule=compare_centre_rule)
        for field_name in ("frames", "detected_frames", "skipped_frames",
                           "raw_violation_detections", "alerts",
                           "suppressed_by_dwell", "zone_hits_foot",
                           "zone_hits_centre"):
            setattr(total, field_name,
                    getattr(total, field_name) + getattr(st, field_name))
        total.modelled_ms += st.modelled_ms

    total.suppressed_by_cooldown = sink.suppressed
    total.fired = list(sink.fired)
    return total
