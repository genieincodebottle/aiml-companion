"""The config file has to be the thing the code reads.

`configs/base.yaml` sat in this repository looking authoritative while the
zone polygons were constants in another module. Editing it changed nothing.
These tests exist so that cannot come back.
"""
from __future__ import annotations

import pytest

from src import site_config
from src.pipeline import run
from src.schemas import PPEClass
from src.site_config import (DEFAULT_CAMERAS, DEFAULT_ZONES, load_cameras,
                             load_zones, zone_camera)
from src.zones import ZoneSet
from tests.conftest import box

YAML = pytest.importorskip("yaml")


def test_the_shipped_config_matches_the_fallback_defaults():
    """Two sources for one value means they drift. This is the gate on that."""
    assert {k: [tuple(p) for p in v] for k, v in load_zones().items()} == \
        {k: [tuple(p) for p in v] for k, v in DEFAULT_ZONES.items()}
    assert [c["id"] for c in load_cameras()] == [c["id"] for c in DEFAULT_CAMERAS]


def test_editing_the_config_changes_the_zones(tmp_path):
    """The test that would have caught the original defect."""
    cfg = tmp_path / "base.yaml"
    cfg.write_text(
        "zones:\n"
        "  loading_bay:\n"
        "    - [0.0, 0.0]\n"
        "    - [100.0, 0.0]\n"
        "    - [100.0, 100.0]\n"
        "    - [0.0, 100.0]\n"
        "cameras:\n"
        "  - id: cam_east\n"
        "    url: rtsp://site/east\n"
        "    zones: [loading_bay]\n",
        encoding="utf-8")

    zones = load_zones(cfg)
    assert list(zones) == ["loading_bay"]
    assert zone_camera(cfg) == {"loading_bay": "cam_east"}
    assert ZoneSet(zones).zone_of(box(50.0, 90.0, height=40.0)) == "loading_bay"


def test_a_missing_config_falls_back_rather_than_crashing(tmp_path):
    assert load_zones(tmp_path / "nope.yaml") == \
        {k: list(v) for k, v in DEFAULT_ZONES.items()}


def test_a_third_zone_clears_its_own_dwell_clock(tmp_path):
    """The hardcoded-names bug, pinned.

    The pipeline used to clear the dwell clock for "crane_radius" and
    "excavation" by name. A third zone kept a stale entry time forever, so a
    worker who left and came back alerted immediately instead of serving the
    dwell again. Nothing errored, and the alert looked correct.
    """
    from src.alerts import AlertSink
    from src.clip_buffer import ClipBuffer, ClipStore
    from src.dwell import DwellTimer
    from src.sim.scene import Frame
    from src.track_smoothing import ClassVote
    from src.tracker import ByteTracker

    zones = {"loading_bay": [(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)]}
    zs = ZoneSet(zones)

    # The worker must WALK out, not teleport. A 600 pixel jump gives the
    # tracker nothing to associate, so it starts a second track and the test
    # measures track churn rather than the dwell clock. Feet at 390 are inside
    # the bay, feet at 410 are outside it, and the boxes still overlap.
    frames = []
    idx = 0

    def at(y, n):
        nonlocal idx
        for _ in range(n):
            frames.append(Frame(idx=idx, ts=idx / 15, camera_id="cam_east",
                                boxes=[box(200.0, y, height=120.0,
                                           cls=PPEClass.NO_HELMET)]))
            idx += 1

    at(390.0, 30)      # 2 s inside, short of the 3 s dwell
    at(410.0, 6)       # steps outside
    at(390.0, 30)      # back in, 2 s again

    dwell = DwellTimer()
    sink = AlertSink(ClipBuffer(15, 7), ClipStore(str(tmp_path)))
    run(frames, lambda f: f.boxes, ByteTracker(high_conf=0.45), ClassVote(),
        zs, dwell, sink, conf=0.45)

    assert sink.fired == [], "the dwell clock survived the worker leaving"
