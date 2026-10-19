"""The knee: where adding concurrency stops buying throughput and starts buying latency.
YOU implement find_knee and plot_knee.
"""
from __future__ import annotations


def find_knee(points: list, min_gain: float = 0.15) -> int:
    """YOU. `points` is [(concurrency, output_tok_per_s), ...] sorted by concurrency.

    Return the concurrency of the first point whose NEXT point gains less than `min_gain`
    (default 15 percent) in throughput. If throughput keeps rising by more than that all the
    way up, return the last concurrency. Raise ValueError on an empty list. A single point
    returns that point.

    Example: [(1, 100), (2, 190), (4, 340), (8, 400), (16, 410)] -> 8
    (4 -> 8 gains 17.6 percent, 8 -> 16 gains 2.5 percent, so the knee is 8).
    """
    raise NotImplementedError("analysis/knee.py: find_knee")


def plot_knee(points: list, out_path: str, knee: int | None = None, title: str = "") -> None:
    """YOU. Write a PNG of the throughput-versus-latency curve.

    `points` is a list of {"concurrency": int, "output_tok_per_s": float, "p99_ttft_s": float,
    "p99_itl_s": float}. Put output tokens per second on the x axis and p99 latency on the y axis
    (two lines: TTFT and ITL, or two panels), label each point with its concurrency, and mark
    the knee. Use the non-interactive backend (matplotlib.use("Agg")) so it works headless.
    Label both axes with units. The test checks the file is a PNG of reasonable size, nothing
    more, so the picture is judged by you and by the person reading your report.
    """
    raise NotImplementedError("analysis/knee.py: plot_knee")
