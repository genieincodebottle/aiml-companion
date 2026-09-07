"""Dwell, cooldown, and the reduction they produce together."""
from __future__ import annotations

import pytest

from src.alerts import AlertSink
from src.clip_buffer import ClipBuffer, ClipStore
from src.dwell import DwellTimer
from src.schemas import Box, PPEClass, Track
from tests.conftest import box


def _track(tid=1, ts=0.0):
    return Track(track_id=tid, box=box(300.0, 430.0), first_seen_ts=ts)


def test_a_brief_crossing_does_not_alert():
    d = DwellTimer(dwell_seconds=3.0)
    assert not d.should_alert(1, "crane_radius", 0.0)
    assert not d.should_alert(1, "crane_radius", 2.9)


def test_holding_the_zone_past_the_threshold_alerts_once():
    d = DwellTimer(dwell_seconds=3.0)
    d.should_alert(1, "crane_radius", 0.0)
    assert d.should_alert(1, "crane_radius", 3.0)
    # One alert per track per zone, not one a second afterwards.
    assert not d.should_alert(1, "crane_radius", 4.0)
    assert not d.should_alert(1, "crane_radius", 60.0)


def test_leaving_restarts_the_clock():
    d = DwellTimer(dwell_seconds=3.0)
    d.should_alert(1, "crane_radius", 0.0)
    d.left(1, "crane_radius")
    assert not d.should_alert(1, "crane_radius", 2.0)   # clock restarted
    assert d.should_alert(1, "crane_radius", 5.0)


def test_cooldown_keys_on_zone_and_violation_not_the_track(tmp_path):
    """A group entering one zone is one situation and one response."""
    sink = AlertSink(ClipBuffer(15, 7), ClipStore(str(tmp_path)),
                     cooldown_s=300.0)
    a = sink.fire(_track(1), "crane_radius", PPEClass.NO_HELMET, "cam_north", 0.0)
    b = sink.fire(_track(2), "crane_radius", PPEClass.NO_HELMET, "cam_north", 10.0)
    assert a is not None
    assert b is None                    # same zone, same violation, suppressed
    assert sink.suppressed == 1


def test_a_different_violation_in_the_same_zone_still_reports(tmp_path):
    sink = AlertSink(ClipBuffer(15, 7), ClipStore(str(tmp_path)),
                     cooldown_s=300.0)
    sink.fire(_track(1), "crane_radius", PPEClass.NO_HELMET, "cam_north", 0.0)
    b = sink.fire(_track(2), "crane_radius", PPEClass.NO_VEST, "cam_north", 10.0)
    assert b is not None


def test_the_clip_covers_the_seconds_before_the_alert(tmp_path):
    """The approach is the part that shows intent, not the aftermath."""
    buf = ClipBuffer(fps=15, seconds=7)
    for i in range(200):
        buf.push(object(), ts=i / 15)
    snap = buf.snapshot()
    assert len(snap) == 105                     # 7 s at 15 fps, capped
    assert snap[-1][1] == pytest.approx(199 / 15)
    assert snap[0][1] == pytest.approx(95 / 15)  # earlier than the alert
