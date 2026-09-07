"""Logging. Replaces print() so pipeline output is greppable in a scheduler."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
FMT = "%(asctime)s %(levelname)-7s %(name)-30s %(message)s"
LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def configure(level: str | None = None) -> None:
    """Set up logging once, but always honour an explicit level.

    The early-return used to be unconditional, and because every module calls
    `get_logger` at import time, logging was already configured before the CLI
    ever parsed its arguments. `--log-level DEBUG` therefore did nothing at all
    while still appearing in `--help`. An explicit level now reconfigures.
    """
    global _CONFIGURED
    if _CONFIGURED and level is None:
        return
    if level is not None:
        wanted = level.upper()
        if wanted not in LEVELS:
            raise ValueError(
                f"unknown log level {level!r}; choose one of {', '.join(LEVELS)}")
    else:
        wanted = os.environ.get("CVTRAPS_LOG_LEVEL", "INFO").upper()
        if wanted not in LEVELS:
            wanted = "INFO"
    logging.basicConfig(level=wanted, format=FMT, datefmt="%H:%M:%S",
                        stream=sys.stdout, force=True)
    for noisy in ("matplotlib", "numexpr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
