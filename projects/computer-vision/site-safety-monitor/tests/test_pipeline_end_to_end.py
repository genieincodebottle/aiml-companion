"""The gate that would catch a regression in the parts that matter.

Every other test in this repository checks one component against a case
written by hand. This one runs the whole chain over a simulated half hour
and pins the numbers the README quotes, so a change that quietly breaks
tracking or dwell fails here rather than being discovered by reading the
demo output and finding it plausible.

Three hours, not the full shift, and not less. A shift takes about a
minute, which is too slow for a pre-commit gate. Half an hour scales down
to three ground-truth violations, and precision over three events moves
in steps of 0.33, so the thresholds below would fire on noise rather than
on a regression. Three hours gives nineteen events, which is the point
where the numbers hold still between seeds.
"""
from __future__ import annotations

import pytest

from src.alerts import AlertSink
from src.clip_buffer import ClipBuffer, ClipStore
from src.config import settings
from src.eval.alerts import alert_metrics
from src.eval.threshold_sweep import sweep
from src.pipeline import run_shift
from src.sim.scene import ZONES, generate

HOURS = 3.0


@pytest.fixture(scope="module")
def shift():
    return generate(hours=HOURS)


@pytest.fixture(scope="module")
def result(shift, tmp_path_factory):
    frames, events = shift
    store = ClipStore(str(tmp_path_factory.mktemp("clips")))
    sink = AlertSink(ClipBuffer(settings.fps, settings.clip_seconds), store)
    st = run_shift(frames, lambda f: f.boxes, ZONES, sink,
                   compare_centre_rule=True)
    return st, alert_metrics(st.fired, events), events


def test_the_scene_is_trackable_at_all(shift):
    """The regression that started this file.

    An early scene placed actors at a random point each frame. The
    pipeline ran and produced plausible numbers while the tracker was
    associating different people into one track, and a 22 second
    violation with 376 correct detections produced no alert.
    """
    frames, events = shift
    assert len(events) >= 15
    assert sum(len(v) for v in frames.values()) > 10_000


def test_alerts_are_far_fewer_than_raw_detections(result):
    """The reduction is the product, and none of it comes from the detector."""
    st, _m, _e = result
    assert st.raw_violation_detections > 800
    assert st.alerts < st.raw_violation_detections / 20
    assert st.reduction_ratio > 20


def test_precision_and_recall_at_the_operating_point(result):
    st, m, events = result
    assert m["precision"] >= 0.80, f"precision fell to {m['precision']:.2f}"
    assert m["recall"] >= 0.70, f"recall fell to {m['recall']:.2f}"
    assert m["alerts"] <= len(events) * 1.5


def test_the_feet_rule_removes_a_large_share_of_zone_hits(result):
    """If this narrows, someone has changed zone_of back to the box centre."""
    st, _m, _e = result
    assert st.zone_hits_centre > st.zone_hits_foot * 1.5


def test_raising_the_threshold_trades_recall_for_precision(shift,
                                                           tmp_path_factory):
    """A flat sweep means the configured threshold is not reaching the
    tracker. That has happened here once already."""
    frames, events = shift
    store = ClipStore(str(tmp_path_factory.mktemp("sweep")))
    rows = sweep(frames, events, ZONES, store, lambda f: f.boxes,
                 thresholds=(0.15, 0.45))
    low, high = rows
    assert high["precision"] > low["precision"] + 0.05
    assert high["false_per_shift"] < low["false_per_shift"]


def test_the_modelled_frame_still_fits(result):
    st, _m, _e = result
    assert st.modelled_avg_ms == pytest.approx(22.50, abs=0.5)
    assert st.modelled_avg_ms < settings.frame_budget_ms
