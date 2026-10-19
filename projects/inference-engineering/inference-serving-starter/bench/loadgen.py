"""Load generators. YOU implement all four functions.

`send` is an async callable you are handed: `await send(item) -> RawResult`. In the real
benchmark it wraps bench.client.stream_completion; in the tests it is a fake. Your code
must not care which.

Why two generators: they answer different questions and they fail differently.

  open loop    requests arrive on a schedule (Poisson at a target QPS) whether or not the
               server keeps up. This is what production traffic does. When the server falls
               behind, the queue grows and latency explodes, which is the truth.
  closed loop  N clients each send the next request when the previous one finishes. The
               server can never be overloaded past N in flight, so latency saturates and a
               closed loop quietly hides the collapse an open loop shows. It is the right
               tool for finding the throughput ceiling, the wrong one for a latency SLO.
"""
from __future__ import annotations

import random
from typing import Awaitable, Callable

from .clock import SimClock
from .dataset import WorkItem

Send = Callable[[WorkItem], Awaitable["RawResult"]]  # noqa: F821


def poisson_arrival_times(qps: float, duration_s: float, rng: random.Random) -> list:
    """YOU. Arrival times (sim seconds from 0) of a Poisson process with rate `qps` over
    [0, duration_s). Inter-arrival gaps are exponential with mean 1/qps.

    Contract: sorted ascending, all < duration_s, deterministic for a seeded rng, and over a
    long horizon the mean gap is 1/qps and the gap standard deviation is also about 1/qps
    (a uniform or fixed-interval schedule is NOT Poisson and the test rejects it).
    """
    raise NotImplementedError("bench/loadgen.py: poisson_arrival_times")


async def run_open_loop(send: Send, items: list, arrival_times: list, clock: SimClock) -> list:
    """YOU. Send items[i] at clock time arrival_times[i], WITHOUT waiting for earlier
    requests to finish. Return the RawResults in item order after all have completed.

    Contract: request i is issued within about 1 sim second of arrival_times[i] even when
    earlier requests are still in flight (the test uses a slow fake `send` and checks that
    the schedule is kept and that more than one request is in flight at once).
    items and arrival_times must be the same length; raise ValueError if not.
    """
    raise NotImplementedError("bench/loadgen.py: run_open_loop")


async def run_closed_loop(send: Send, items: list, concurrency: int) -> list:
    """YOU. Run `concurrency` workers that each pull the next unsent item, send it, wait for
    the result, then pull again, until every item has been sent exactly once.
    Return the RawResults in item order.

    Contract: never more than `concurrency` requests in flight; exactly len(items) sent;
    concurrency < 1 raises ValueError.
    """
    raise NotImplementedError("bench/loadgen.py: run_closed_loop")


async def concurrency_sweep(make_send: Callable[[], Send], items_for: Callable[[int], list],
                            levels: list) -> dict:
    """YOU. Closed-loop run at each concurrency level, lowest first, and return
    {level: [RawResult, ...]}.

    * `items_for(level)` gives the WorkItems for that level (more requests at higher
      concurrency, so every level has a steady-state window).
    * Call make_send() once per level to get a fresh `send`.
    * Levels run one after another, never together.
    """
    raise NotImplementedError("bench/loadgen.py: concurrency_sweep")
