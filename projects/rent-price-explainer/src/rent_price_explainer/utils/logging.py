"""Logging. Replaces print() so pipeline output is greppable in a scheduler."""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
FMT = "%(asctime)s %(levelname)-7s %(name)-38s %(message)s"


def configure(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or os.environ.get("RENT_LOG_LEVEL", "INFO")).upper(),
        format=FMT, datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    for noisy in ("matplotlib", "numexpr", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
