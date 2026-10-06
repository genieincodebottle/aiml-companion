"""Injectable clock. Pipeline code asks the clock, never `date.today()`.

The wall-clock mistake in `naive.py` needs a deterministic "today", so a test or
a demo hands the pipeline a `FrozenClock`.
"""
from datetime import date


class SystemClock:
    def today(self):
        return date.today()


class FrozenClock:
    def __init__(self, today):
        self._today = today

    def today(self):
        return self._today
