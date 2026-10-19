"""Fleet sizing with Little's law. YOU implement all three functions.

Little's law: the average number of requests in the system L equals the arrival rate lambda
times the average time each spends there W. So at peak, a service taking `peak_qps` requests a
second that each live `mean_e2e_s` seconds holds L = peak_qps * mean_e2e_s requests in flight.
One replica can hold only so many before it breaks the SLO (that is the number you got from
analysis.goodput.max_slo_concurrency). The ratio is the replica count, plus headroom.
"""
from __future__ import annotations


def concurrent_requests(peak_qps: float, mean_e2e_s: float) -> float:
    """YOU. L = lambda * W. Raise ValueError if either input is negative."""
    raise NotImplementedError("analysis/fleet.py: concurrent_requests")


def replicas_needed(peak_qps: float, mean_e2e_s: float, max_concurrency_per_replica: int,
                    headroom: float = 0.30) -> int:
    """YOU. Smallest whole number of replicas such that the in-flight load, at each replica
    running at most (1 - headroom) of its SLO-safe concurrency, fits:

        replicas = ceil( L / (max_concurrency_per_replica * (1 - headroom)) )

    At least 1. Raise ValueError when max_concurrency_per_replica < 1 or headroom is outside
    [0, 1).

    Example: 10 qps, 4 s each, 16 per replica, 25 percent headroom -> L = 40, 40 / 12 = 3.33,
    so 4 replicas.
    """
    raise NotImplementedError("analysis/fleet.py: replicas_needed")


def fleet_plan(peak_qps: float, mean_e2e_s: float, max_concurrency_per_replica: int,
               gpus_per_replica: int = 1, headroom: float = 0.30) -> dict:
    """YOU. {"in_flight": L, "replicas": n, "gpus": n * gpus_per_replica,
    "headroom": headroom, "utilisation_at_peak": L / (n * max_concurrency_per_replica)}."""
    raise NotImplementedError("analysis/fleet.py: fleet_plan")
