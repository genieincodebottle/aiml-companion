"""A clock in simulated seconds. GIVEN.

engine_sim runs `time_scale` times faster than wall time, so a load generator that wants
"2 requests per second of engine time" has to schedule in sim seconds. Against a real
vLLM use SimClock(1.0) and sim seconds are just seconds.
"""
from __future__ import annotations

import asyncio
import sys
import time


def _raise_timer_resolution() -> None:
    """Windows sleeps in 15.6 ms steps unless asked for 1 ms; at time_scale 10 that is 0.16 sim s."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


class SimClock:
    def __init__(self, time_scale: float = 1.0):
        _raise_timer_resolution()
        self.time_scale = float(time_scale)
        self._t0 = time.monotonic()

    def now(self) -> float:
        """Simulated seconds since the clock was created."""
        return (time.monotonic() - self._t0) * self.time_scale

    async def sleep_until(self, sim_t: float) -> None:
        """Sleep until the clock reads `sim_t`. Returns immediately if that time has passed."""
        delay = (sim_t - self.now()) / self.time_scale
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)

    async def sleep(self, sim_seconds: float) -> None:
        await self.sleep_until(self.now() + sim_seconds)
