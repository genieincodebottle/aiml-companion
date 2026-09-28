"""Structured, OTel-shaped tracing for every agent run.

Every meaningful step - a tool call, an edit, a test run, a guardrail block - is a
span. Without this you cannot debug a 12-step trajectory across a fleet of repos,
and you cannot prove the digital employee is getting better over time.

We keep spans in memory and append them as JSONL under the data dir. If the optional
`opentelemetry` package is installed we also emit real OTel spans; if not, the JSONL
sink is the source of truth. This is the "wiring point" - swap the JSONL sink for an
OTLP exporter to ship to Honeycomb / Grafana / Langfuse without touching call sites.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace as _otel_trace

    _OTEL_TRACER = _otel_trace.get_tracer("migration-engineer")
except Exception:  # pragma: no cover
    _OTEL_TRACER = None


class Tracer:
    """A tiny span recorder. One instance per job."""

    def __init__(self, job_id: str, data_dir: Path) -> None:
        self.job_id = job_id
        self.spans: list[dict[str, Any]] = []
        self._sink = data_dir / "traces"
        self._sink.mkdir(parents=True, exist_ok=True)
        self._path = self._sink / f"{job_id}.jsonl"

    @contextmanager
    def span(self, name: str, **attrs: Any):
        start = time.time()
        record: dict[str, Any] = {"job_id": self.job_id, "name": name, "attrs": attrs, "ts": start}
        # start_span (NOT start_as_current_span): callers await inside this span while
        # the event loop interleaves other repos' tasks, so attaching it to the ambient
        # context would mis-parent concurrent traces and detach out of order.
        otel_span = _OTEL_TRACER.start_span(name) if _OTEL_TRACER else None
        if otel_span is not None:  # pragma: no cover - only when OTel installed
            for k, v in attrs.items():
                otel_span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        try:
            yield record
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            record["status"] = "error"
            record["error"] = str(exc)
            if otel_span is not None:  # pragma: no cover
                otel_span.record_exception(exc)
                otel_span.set_attribute("error", True)
            raise
        finally:
            record["duration_ms"] = round((time.time() - start) * 1000, 2)
            self.spans.append(record)
            self._append(record)
            if otel_span is not None:  # pragma: no cover
                otel_span.end()

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:  # pragma: no cover - tracing must never break the run
            pass
