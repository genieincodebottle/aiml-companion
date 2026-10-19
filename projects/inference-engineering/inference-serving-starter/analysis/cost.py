"""Cost per million tokens, from MEASURED throughput. YOU implement it.

There is deliberately no price table in this repo. GPU prices change weekly and differ by
cloud, region and commitment. You supply `gpu_hourly_usd` from the provider page you
rented from, and your report names that page and the date you read it.
"""
from __future__ import annotations


def cost_per_million_tokens(tokens_per_s: float, gpu_hourly_usd: float, num_gpus: int = 1,
                            utilisation: float = 1.0) -> float:
    """YOU. Dollars to produce one million tokens.

        cost = (gpu_hourly_usd * num_gpus) / (tokens_per_s * 3600 * utilisation) * 1e6

    `utilisation` (0 < u <= 1) is the fraction of the hour the GPUs are doing useful work at
    the measured rate; a fleet sized for peak is idle most of the day, and that idle time is
    part of the real cost. Raise ValueError when tokens_per_s <= 0, utilisation is outside
    (0, 1], or gpu_hourly_usd < 0.

    Example: $2.00/h, 1000 tok/s, 1 GPU, full utilisation -> about $0.556 per million tokens.
    """
    raise NotImplementedError("analysis/cost.py: cost_per_million_tokens")
