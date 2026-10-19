"""Run engine_sim in pure virtual time and emit what an operator would have on file after
an incident: Prometheus scrapes taken at a fixed interval, and the engine's periodic log
lines. No server, no wall clock, byte-for-byte reproducible.
"""
from __future__ import annotations

from typing import Callable, Optional

from .config import EngineConfig
from .core import Engine, Request

LOG_EVERY_S = 10.0
SCRAPE_EVERY_S = 30.0


def _stamp(t: float) -> str:
    # A fixed fictional start (2026-10-19 02:00:00 UTC) so logs look like logs.
    s = int(t)
    return f"2026-10-19 {2 + s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def simulate(cfg: EngineConfig, requests: list, until: float, *, scrape_every: float = SCRAPE_EVERY_S,
             log_every: float = LOG_EVERY_S, accept_fn: Optional[Callable[[float], float]] = None,
             engine_id: str = "0", label: str = "engine") -> dict:
    eng = Engine(cfg, accept_fn)
    for r in requests:
        eng.submit(Request(r.rid, r.arrival, r.prompt_ids, r.max_tokens, r.tag))
    scrapes: list = []
    logs: list = []
    next_scrape, next_log = scrape_every, log_every
    prev = {"p": 0, "g": 0, "q": 0, "h": 0, "pre": 0}
    t = 0.0
    while t < until - 1e-9:
        t = round(t + 1.0, 6)
        eng.run_until(t)
        if t >= next_log - 1e-9:
            m = eng.metrics
            dt = log_every
            q = m.prefix_queries - prev["q"]
            h = m.prefix_hits - prev["h"]
            hit = 100.0 * h / q if q else 0.0
            logs.append(
                f"INFO {_stamp(t)} [engine_sim:{label}] Avg prompt throughput: {(m.prompt_tokens - prev['p']) / dt:.1f} "
                f"tokens/s, Avg generation throughput: {(m.generation_tokens - prev['g']) / dt:.1f} tokens/s, "
                f"Running: {len(eng.running)} reqs, Waiting: {len(eng.waiting)} reqs, "
                f"KV cache usage: {eng.kv_usage() * 100:.1f}%, Prefix cache hit rate: {hit:.1f}%")
            pre = m.preemptions - prev["pre"]
            if pre > 0:
                logs.append(
                    f"WARNING {_stamp(t)} [engine_sim:{label}] {pre} sequences preempted in the last "
                    f"{dt:.0f}s: KV cache is full, their blocks were freed and they will be recomputed")
            prev = {"p": m.prompt_tokens, "g": m.generation_tokens, "q": m.prefix_queries,
                    "h": m.prefix_hits, "pre": m.preemptions}
            next_log += log_every
        if t >= next_scrape - 1e-9:
            scrapes.append((t, eng.render_metrics(engine_id)))
            next_scrape += scrape_every
    return {"scrapes": scrapes, "logs": logs, "engine": eng}
