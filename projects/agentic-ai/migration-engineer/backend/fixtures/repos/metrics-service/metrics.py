"""Epoch-to-datetime helpers for the metrics-service.

EXERCISE FIXTURE (see docs/EXERCISES.md #2): this module uses the deprecated
datetime.utcfromtimestamp(), which returns a NAIVE datetime. No rule in the rulebook
migrates it yet - writing that rule end-to-end is the exercise. `timezone` is not
imported, so a correct migration needs the two-pass fix.
"""

from datetime import datetime


def event_from_epoch(ts: float) -> datetime:
    return datetime.utcfromtimestamp(ts)


def window_start(ts: float, seconds: int = 60) -> datetime:
    return datetime.utcfromtimestamp(ts - (ts % seconds))
