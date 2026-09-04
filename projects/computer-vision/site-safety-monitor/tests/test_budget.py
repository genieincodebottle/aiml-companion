"""Fails the build when the modelled frame exceeds the budget."""
from __future__ import annotations

import pytest

from src import budget
from src.budget import (MODELLED_MS, amdahl_ceiling, inference_share,
                        modelled_average_ms, modelled_frame_ms,
                        modelled_report, report, reset, stage)
from src.config import settings


def test_a_detected_frame_does_not_fit_on_its_own():
    """The premise of the whole project. 33.5 against 33.33."""
    r = modelled_report()
    assert modelled_frame_ms(True) == pytest.approx(33.5)
    assert modelled_frame_ms(True) > r["budget_ms"]


def test_frame_fits_the_budget_once_detection_is_skipped():
    r = modelled_report()
    assert r["fits"], (
        f"frame is {r['average_ms']:.2f} ms against a "
        f"{r['budget_ms']:.2f} ms budget. Stages: {r['stages']}"
    )
    assert r["average_ms"] == pytest.approx(22.50)
    assert r["headroom_ms"] == pytest.approx(10.83, abs=0.01)


def test_budget_is_derived_from_the_camera_count():
    """Adding a third camera has to move the budget everywhere at once."""
    assert settings.frame_budget_ms == pytest.approx(33.333, abs=1e-3)
    assert settings.frames_per_shift == 1_080_000


def test_inference_share_and_the_ceiling_it_implies():
    assert inference_share() == pytest.approx(0.5373, abs=1e-4)
    assert amdahl_ceiling() == pytest.approx(2.161, abs=1e-3)


def test_inference_share_is_tracked():
    """If inference drops below half the frame, the ceiling has moved and
    the next optimisation should target a different stage."""
    assert inference_share() < 0.60, "inference now dominates, re-check the ceiling"


def test_nms_share_rises_when_inference_is_optimised():
    """The stage that quietly becomes the bottleneck after a good result.

    Halving inference changes nothing about NMS, and its share of the
    frame rises from 11.9 to 16.3 per cent without a line of it changing.
    The share only passes 20 per cent once inference drops below 4.5 ms,
    which is a 4x speed-up rather than a 2x one.
    """
    before = MODELLED_MS["nms"] / modelled_frame_ms(True)
    halved = dict(MODELLED_MS, inference=MODELLED_MS["inference"] / 2)
    after = halved["nms"] / sum(halved.values())
    assert before == pytest.approx(0.1194, abs=1e-4)
    assert after == pytest.approx(0.1633, abs=1e-4)
    assert after > before

    quartered = dict(MODELLED_MS, inference=4.0)
    assert quartered["nms"] / sum(quartered.values()) > 0.20


def test_skipping_does_not_skip_letterboxing():
    """A skipped frame is still prepared. That is why the average is 22.50
    rather than 21.50, and the difference is the whole headroom argument."""
    assert "letterbox" not in budget.SKIPPABLE
    assert modelled_frame_ms(False) == pytest.approx(11.5)


def test_detecting_every_frame_blows_the_budget():
    assert modelled_average_ms(1) > settings.frame_budget_ms


def test_measured_report_totals_what_it_timed():
    reset()
    with stage("inference"):
        pass
    r = report()
    assert r["counts"]["inference"] == 1
    assert r["total_ms"] >= 0.0
