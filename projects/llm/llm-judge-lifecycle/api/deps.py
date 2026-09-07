"""Dependency wiring, and the one place HTTP status codes are chosen.

Services raise plain Python exceptions - ``KeyError`` for an unknown criterion,
``FileNotFoundError`` for a missing week, ``BudgetExceeded`` when a run hits its
cap. Translating those into status codes belongs here and nowhere else, because
a service that knew about 404s could only ever be called from a web request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import HTTPException

from src.runtime import BudgetExceeded
from src.services import (
    BenchmarkService,
    MonitoringService,
    ServingService,
    TuningService,
)
from src.services._context import build_context

# One context per process. The rubric store is on disk, so a promoted rubric is
# picked up by the next request without a restart; what is cached here is the
# config, the domain and the provider clients.
#
# OFFLINE=1 runs the whole API on the rule engine - useful for a demo, a CI
# smoke test, or reading the UI without spending anything.
@lru_cache(maxsize=1)
def get_context():
    return build_context(offline=os.environ.get("OFFLINE") == "1")


def benchmark_service() -> BenchmarkService:
    return BenchmarkService(get_context())


def tuning_service() -> TuningService:
    return TuningService(get_context())


def serving_service() -> ServingService:
    return ServingService(get_context())


def monitoring_service() -> MonitoringService:
    return MonitoringService(get_context())


def run(fn, *args, **kwargs) -> dict[str, Any]:
    """Call a service and map its exceptions to status codes.

    Every handler goes through this, so the mapping exists once. A 402 for a
    budget stop rather than a 500 is deliberate: it is not a server fault, it is
    the caller asking for more work than the run is allowed to pay for, and the
    message says how much was spent and on which role.
    """
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
