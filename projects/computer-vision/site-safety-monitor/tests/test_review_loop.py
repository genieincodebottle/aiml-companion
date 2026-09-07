"""The review loop, and the two reasons that must not become training data."""
from __future__ import annotations

from api.review import decide
from src.monitor.health import Health
from src.review.capture import evaluation_safe, load, record
from src.review.hard_negatives import to_training_example, triage
from src.schemas import Alert, PPEClass
from scripts.test_walk import verify


def _alert(zone="crane_radius", ts=10.0):
    return Alert(track_id=1, camera_id="cam_north",
                 violation=PPEClass.NO_HELMET, zone=zone, dwell_s=3.2,
                 clip_path="artifacts/clips/track_1.json", start_ts=6.0, ts=ts)


def test_a_dismissal_needs_a_reason():
    assert "error" in decide("a1", confirmed=False, reason=None)


def test_an_unknown_reason_is_refused():
    assert "error" in decide("a1", confirmed=False, reason="because")


def test_mined_rows_never_reach_the_evaluation_set(tmp_path):
    """They were selected because the pipeline was unsure, so they are
    harder than site traffic by construction."""
    p = tmp_path / "reviewed.jsonl"
    record("a1", confirmed=False, reason="wrong_detection", path=p)
    record("a2", confirmed=True, reason=None, sampling="random", path=p)
    rows = load(p)
    assert len(rows) == 2
    assert len(evaluation_safe(rows)) == 1


def test_only_a_wrong_detection_becomes_a_training_example():
    a = _alert()
    assert to_training_example(a, "wrong_detection")["weight"] == 2.0
    # A worker legitimately inside a zone is a polygon problem. Training on
    # it teaches the model to miss real people in exactly that place.
    assert to_training_example(a, "outside_zone") is None
    assert to_training_example(a, "authorised") is None


def test_triage_groups_dismissals_by_what_has_to_change():
    rows = [
        {"confirmed": False, "reason": "wrong_detection"},
        {"confirmed": False, "reason": "wrong_detection"},
        {"confirmed": False, "reason": "outside_zone"},
        {"confirmed": False, "reason": "authorised"},
        {"confirmed": True, "reason": None},
    ]
    t = triage(rows)
    assert t == {"retrain": 2, "fix_polygon": 1, "fix_rule": 1, "unknown": 0}


def test_health_reports_the_silent_failures():
    """Zero alerts is indistinguishable from good news, so count detections."""
    h = Health()
    assert h.check(last_frame_ts=1000.0, alerts_24h=3, detections_1h=900,
                   now=1005.0) == []

    dead = h.check(last_frame_ts=1000.0, alerts_24h=0, detections_1h=0,
                   now=1100.0)
    assert len(dead) == 3

    # A site can genuinely have no violations. It cannot have no people.
    quiet = h.check(last_frame_ts=1000.0, alerts_24h=0, detections_1h=900,
                    now=1005.0)
    assert quiet == ["no alerts in 24 h, verify with a test walk"]


def test_the_scripted_walk_checks_the_physical_half():
    assert verify("crane_radius", started_at=0.0, alerts=[_alert(ts=6.0)])
    # Too late: dwell 3 s plus a 5 s margin.
    assert not verify("crane_radius", started_at=0.0, alerts=[_alert(ts=20.0)])
    assert not verify("excavation", started_at=0.0, alerts=[_alert(ts=6.0)])
