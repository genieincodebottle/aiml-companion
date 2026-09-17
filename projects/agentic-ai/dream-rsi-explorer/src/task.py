"""The discovery task: pack 26 circles in a unit square, maximise the sum of radii.

This is one of the mathematical optimisation benchmarks in the Dream-RSI paper
(and in AlphaEvolve before it). It is a good teaching task for three reasons.
The evaluator is exact and costs microseconds. A candidate is plain data, so an
LLM can propose one as JSON. And progress is gradual, so an exploration policy
has real decisions to make about where to spend attempts.

For reference, the best published sum for 26 circles is about 2.635. The point
here is to compare exploration policies on the same model, not to set a record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

N_CIRCLES = 26
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Evaluation:
    score: float
    valid: bool
    diagnostics: str


def evaluate(layout: dict) -> Evaluation:
    """Score a candidate. Invalid layouts score 0.0 and say why.

    The diagnostics string matters as much as the score. It is stored on the
    tree node, and a live discovery agent reads it when it builds on that node.
    """
    try:
        centers = [(float(x), float(y)) for x, y in layout["centers"]]
        radii = [float(r) for r in layout["radii"]]
    except (KeyError, TypeError, ValueError) as exc:
        return Evaluation(0.0, False, f"malformed layout: {exc}")

    if len(centers) != N_CIRCLES or len(radii) != N_CIRCLES:
        return Evaluation(
            0.0, False, f"expected {N_CIRCLES} circles, got {len(centers)} centers and {len(radii)} radii"
        )
    if not all(math.isfinite(v) for c in centers for v in c) or not all(math.isfinite(r) for r in radii):
        return Evaluation(0.0, False, "non-finite value")
    if any(r < 0 for r in radii):
        return Evaluation(0.0, False, "negative radius")

    for i, ((x, y), r) in enumerate(zip(centers, radii)):
        if x - r < -TOLERANCE or x + r > 1 + TOLERANCE or y - r < -TOLERANCE or y + r > 1 + TOLERANCE:
            return Evaluation(0.0, False, f"circle {i} crosses the square boundary")

    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            gap = math.dist(centers[i], centers[j]) - radii[i] - radii[j]
            if gap < -TOLERANCE:
                return Evaluation(0.0, False, f"circles {i} and {j} overlap by {-gap:.2e}")

    total = sum(radii)
    smallest = min(range(N_CIRCLES), key=lambda i: radii[i])
    return Evaluation(
        round(total, 6),
        True,
        f"valid, sum of radii {total:.6f}, smallest circle {smallest} has r={radii[smallest]:.4f}",
    )


def fit_radii(centers: list[tuple[float, float]], sweeps: int = 60) -> list[float]:
    """Largest radii we can cheaply find for fixed centres, always valid.

    Starts from all-zero radii, which is feasible, then repeatedly grows each
    circle to the largest radius the square and its neighbours allow. Every step
    keeps the layout feasible, so the result never needs repairing. It is not
    the optimal LP solution, just a good, dependency-free one.
    """
    n = len(centers)
    neighbours = [
        [(j, math.dist(centers[i], centers[j])) for j in range(n) if j != i] for i in range(n)
    ]
    wall = [min(x, 1 - x, y, 1 - y) for x, y in centers]
    radii = [0.0] * n
    for _ in range(sweeps):
        grew = 0.0
        for i in range(n):
            limit = min(wall[i], min(d - radii[j] for j, d in neighbours[i]))
            limit = max(0.0, limit)
            grew = max(grew, limit - radii[i])
            radii[i] = limit
        if grew < 1e-9:
            break
    # A hair of margin so floating-point error never fails the evaluator.
    return [max(0.0, r - 1e-12) for r in radii]

