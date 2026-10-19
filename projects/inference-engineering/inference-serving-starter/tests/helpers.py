"""Closed-loop driver for the engine core in virtual time. GIVEN. Used by the engine tests."""
from __future__ import annotations

import random

from engine_sim.core import Engine, Request


def closed_loop(cfg, concurrency: int, n: int, prompt_len: int = 600, out_len: int = 150, seed: int = 1,
                shared_prefix: int = 0) -> dict:
    """c clients, each sends its next request when the previous finishes. Virtual time."""
    eng = Engine(cfg)
    rng = random.Random(seed)
    prefix = [rng.randrange(1 << 30) for _ in range(shared_prefix)]

    def mk(i: int, t: float) -> Request:
        body = [rng.randrange(1 << 30) for _ in range(prompt_len - shared_prefix)]
        return Request(f"r{i}", t, prefix + body, out_len)

    sent = 0
    for _ in range(min(concurrency, n)):
        eng.submit(mk(sent, 0.0))
        sent += 1
    done, ttft, tokens = 0, {}, 0
    while done < n:
        horizon = eng.now + 1.0 if eng.busy else (eng.pending[0][0] + 1e-9 if eng.pending else None)
        if horizon is None:
            break
        for ev in eng.run_until(horizon):
            tokens += ev.n
            if ev.first:
                ttft[ev.rid] = ev.t - ev.arrival
            if ev.done:
                done += 1
                if sent < n:
                    eng.submit(mk(sent, ev.t))
                    sent += 1
    vals = sorted(ttft.values())
    return {"tok_per_s": tokens / eng.now, "ttft_p50": vals[len(vals) // 2], "ttft": ttft,
            "preemptions": eng.metrics.preemptions, "engine": eng, "elapsed": eng.now}
