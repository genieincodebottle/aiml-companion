"""Deterministic synthetic workloads for engine_sim (used by debug_lab and the tests).

Pure standard library (random.Random), so the same seed gives the same bytes on every
platform. The benchmark harness you write uses bench/ instead; this module exists so the
incident bundles in debug_lab/ can be regenerated exactly.
"""
from __future__ import annotations

import math
import random

from .core import Request


def lognormal_int(rng: random.Random, median: float, sigma: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(rng.lognormvariate(math.log(median), sigma)))))


def poisson_arrivals(segments: list, rng: random.Random) -> list:
    """segments = [(t_start, t_end, qps), ...]. Returns sorted arrival times."""
    out: list = []
    for t0, t1, qps in segments:
        if qps <= 0:
            continue
        t = t0
        while True:
            t += rng.expovariate(qps)
            if t >= t1:
                break
            out.append(t)
    return out


def unique_ids(rng: random.Random, n: int) -> list:
    return [rng.randrange(1 << 30) for _ in range(n)]


def prefix_ids(group: int, n: int) -> list:
    """Token ids of a shared prefix. Same group -> same ids -> prefix-cache hits."""
    r = random.Random(1_000_003 * (group + 1))
    return [r.randrange(1 << 30) for _ in range(n)]


def make_requests(arrivals: list, rng: random.Random, *, prompt_median: int, prompt_sigma: float,
                  out_median: int, out_sigma: float, prompt_range=(16, 3000), out_range=(8, 1000),
                  prefix_groups: list | None = None, tag_prefix: str = "r") -> list:
    """prefix_groups = [(weight, group_id, prefix_tokens), ...] or None for no shared prefix."""
    reqs = []
    for i, t in enumerate(arrivals):
        ids: list = []
        tag = ""
        if prefix_groups:
            r = rng.random() * sum(w for w, _, _ in prefix_groups)
            acc = 0.0
            for w, gid, ptoks in prefix_groups:
                acc += w
                if r <= acc:
                    ids = prefix_ids(gid, ptoks)
                    tag = f"g{gid}"
                    break
        n_unique = lognormal_int(rng, prompt_median, prompt_sigma, prompt_range[0], prompt_range[1])
        ids = ids + unique_ids(rng, n_unique)
        n_out = lognormal_int(rng, out_median, out_sigma, out_range[0], out_range[1])
        reqs.append(Request(f"{tag_prefix}{i}", t, ids, n_out, tag))
    return reqs
