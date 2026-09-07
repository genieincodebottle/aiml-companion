"""Orchestration and policy. The only layer the API and the CLI are allowed to call.

    app/            UI. Renders. Decides nothing.
    api/routes_*    Transport. Validates, calls ONE service, maps the result.
    src/services/   Orchestration and policy.   <- you are here
    src/            Capabilities.

Services never import FastAPI. That single rule is what lets `run.py`, the HTTP
API, the tests and a notebook share one code path - a service that could raise
``HTTPException`` would be callable only from a web request, and everything else
would need its own copy of the orchestration. ``tests/test_layering.py`` fails
the build when the boundary is crossed.
"""

from __future__ import annotations

from .benchmark_service import BenchmarkService
from .monitoring_service import MonitoringService
from .serving_service import ServingService
from .tuning_service import TuningService

__all__ = [
    "BenchmarkService",
    "MonitoringService",
    "ServingService",
    "TuningService",
]
