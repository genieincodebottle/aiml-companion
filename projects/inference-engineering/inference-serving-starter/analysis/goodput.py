"""Goodput: the requests per second that actually met the SLO. YOU implement all four.

Throughput counts every token the server produced. Goodput counts only the requests a user
would call acceptable. A server can double its throughput by queueing everything and
destroy its goodput doing it.

A request MEETS the SLO when it succeeded AND ttft_s <= slo_ttft_s AND tpot_s <= slo_tpot_s
(its average inter-token latency). Both limits are inclusive.
"""
from __future__ import annotations


def meets_slo(rec, slo_ttft_s: float, slo_tpot_s: float) -> bool:
    """YOU. True when rec.ok and rec.ttft_s <= slo_ttft_s and rec.tpot_s <= slo_tpot_s."""
    raise NotImplementedError("analysis/goodput.py: meets_slo")


def slo_attainment(records: list, slo_ttft_s: float, slo_tpot_s: float) -> float:
    """YOU. Fraction of ALL records (failed ones count as misses) that meet the SLO.
    Empty list -> 0.0."""
    raise NotImplementedError("analysis/goodput.py: slo_attainment")


def goodput(records: list, slo_ttft_s: float, slo_tpot_s: float, duration_s: float | None = None) -> float:
    """YOU. Requests per second that meet the SLO. duration_s None means the observed span
    (same definition as analysis.tables.throughput). Never exceeds the request throughput."""
    raise NotImplementedError("analysis/goodput.py: goodput")


def max_slo_concurrency(levels: list, slo_ttft_s: float, slo_tpot_s: float,
                        min_attainment: float = 0.95) -> int | None:
    """YOU. `levels` is a list of {"concurrency": int, "records": [RequestRecord, ...]}
    (warm-up already removed). Return the highest concurrency whose SLO attainment is at
    least `min_attainment`, or None if no level qualifies. This is the number a fleet is
    sized with: how many requests one replica can hold in flight and still be acceptable."""
    raise NotImplementedError("analysis/goodput.py: max_slo_concurrency")
