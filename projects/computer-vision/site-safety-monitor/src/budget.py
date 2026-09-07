"""A timing context manager that records every stage against the budget.

Two tables live here and they answer different questions.

`stage()` measures what this machine actually did, which is what the
budget test asserts on. MODELLED_MS is the edge device's costed frame,
which is what the README and the model card quote. A laptop running a
numpy stand-in engine cannot measure an 18 ms TensorRT inference, so
pretending the measured table is the device's would be a lie that reads
as a result.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

from src.config import settings

# The costed frame on the target device, in milliseconds. Ordered as the
# pipeline runs them.
MODELLED_MS = {
    "capture": 4.0,
    "decode": 3.0,
    "letterbox": 2.0,
    "inference": 18.0,
    "nms": 4.0,
    "track": 1.5,
    "zone": 0.5,
    "emit": 0.5,
}

# The stages a skipped frame does not pay for. Letterboxing is NOT in
# here: the frame is still prepared, because the clip buffer stores
# letterboxed frames and the next detection reuses the result.
SKIPPABLE = ("inference", "nms")

_stages: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _stages[name] += (time.perf_counter() - t0) * 1000.0
        _counts[name] += 1


def reset() -> None:
    _stages.clear()
    _counts.clear()


def report(frames: int = 1) -> dict:
    """Measured stage table, averaged per frame."""
    per_frame = {k: v / max(frames, 1) for k, v in _stages.items()}
    total = sum(per_frame.values())
    return {
        "stages": per_frame,
        "counts": dict(_counts),
        "total_ms": total,
        "budget_ms": settings.frame_budget_ms,
        "fits": total <= settings.frame_budget_ms,
    }


def modelled_frame_ms(detect: bool = True) -> float:
    """Cost of one frame on the device, with or without a detection."""
    return sum(v for k, v in MODELLED_MS.items()
               if detect or k not in SKIPPABLE)


def modelled_average_ms(detect_every: int | None = None) -> float:
    """Average frame cost once one frame in `detect_every` runs the network."""
    n = detect_every or settings.detect_every
    detected = modelled_frame_ms(True)
    skipped = modelled_frame_ms(False)
    return (detected + skipped * (n - 1)) / n


def inference_share(detect: bool = True) -> float:
    return MODELLED_MS["inference"] / modelled_frame_ms(detect)


def amdahl_ceiling(share: float | None = None) -> float:
    """Best possible speed-up if the named share went to zero cost.

    Making the network infinitely fast is the optimisation everyone
    reaches for first, and this is the number that says how much it can
    ever be worth.
    """
    s = inference_share() if share is None else share
    return 1.0 / (1.0 - s)


def modelled_report(detect_every: int | None = None) -> dict:
    n = detect_every or settings.detect_every
    avg = modelled_average_ms(n)
    budget = settings.frame_budget_ms
    return {
        "stages": dict(MODELLED_MS),
        "detected_ms": modelled_frame_ms(True),
        "skipped_ms": modelled_frame_ms(False),
        "detect_every": n,
        "average_ms": avg,
        "budget_ms": budget,
        "headroom_ms": budget - avg,
        "fits": avg <= budget,
        "inference_share": inference_share(),
        "amdahl_ceiling": amdahl_ceiling(),
    }
