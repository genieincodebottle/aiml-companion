"""Compares exported and original outputs on a fixed batch, which catches the
silent half."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.export import parse_log
from src.preprocess import apply_letterbox, letterbox, unletterbox
from src.sim.engine import SimEngine

FIX = Path(__file__).parent / "fixtures"


def _outputs(seed, jitter_boxes=0.0, flip_class=False):
    """Stand-in for the two engines. Boxes may drift, classes may not."""
    rng = np.random.default_rng(seed)
    ref = np.column_stack([
        rng.uniform(0, 600, 20), rng.uniform(0, 600, 20),
        rng.uniform(0, 600, 20), rng.uniform(0, 600, 20),
        rng.uniform(0.5, 1.0, 20), rng.integers(0, 5, 20),
    ])
    got = ref.copy()
    got[:, :4] += rng.normal(0, jitter_boxes, (20, 4))
    if flip_class:
        got[0, 5] = (got[0, 5] + 1) % 5
    return ref, got


def _assert_parity(ref, got):
    """Two assertions with different tolerances, deliberately not one loose one.

    FP16 conversion moves logits slightly. A box shifting by a pixel is
    fine, a class flipping is not, so both are asserted separately.
    """
    np.testing.assert_allclose(got[:, :4], ref[:, :4], atol=1.0)   # boxes, pixels
    assert (got[:, 5] == ref[:, 5]).all(), "exported engine changed a class"


def test_engine_matches_pytorch():
    ref, got = _outputs(0, jitter_boxes=0.2)
    _assert_parity(ref, got)


def test_a_flipped_class_is_caught_even_though_boxes_are_perfect():
    """The failure a single loose tolerance would hide."""
    ref, got = _outputs(1, jitter_boxes=0.0, flip_class=True)
    with pytest.raises(AssertionError, match="changed a class"):
        _assert_parity(ref, got)


def test_export_refuses_a_partitioned_graph():
    info = parse_log((FIX / "trtexec_partitioned.log").read_text())
    assert info["engines"] == 3
    assert info["partitioned"]
    assert info["fallbacks"] == {"ScatterND", "NonZero"}
    # Two boundaries, both directions, 0.546 ms a copy.
    assert info["copy_overhead_ms"] == pytest.approx(2.184, abs=1e-3)


def test_a_clean_export_passes():
    info = parse_log((FIX / "trtexec_clean.log").read_text())
    assert info["engines"] == 1
    assert not info["partitioned"]
    assert info["copy_overhead_ms"] == 0.0


def test_letterbox_round_trips_a_box():
    """Any change to letterboxing invalidates the engine, so pin the geometry."""
    img = np.zeros((480, 640, 3), np.uint8)
    _, meta = letterbox(img)
    box = (100.0, 50.0, 300.0, 400.0)
    assert unletterbox(apply_letterbox(box, meta), meta) == pytest.approx(box)


def test_letterbox_pads_with_the_training_value():
    img = np.zeros((480, 640, 3), np.uint8)
    out, (s, left, top) = letterbox(img)
    assert out.shape == (640, 640, 3)
    assert left == 0 and top == 80          # 640x480 scaled to 640x480, centred
    assert out[0, 0, 0] == 114


def test_sim_engine_is_deterministic():
    a = SimEngine(seed=3).infer(np.zeros((640, 640, 3), np.uint8))
    b = SimEngine(seed=3).infer(np.zeros((640, 640, 3), np.uint8))
    np.testing.assert_array_equal(a, b)
