"""Distributed tracing: one trace per workflow, across every plane.

Every span records `trace_id`, `span_id` and `parent_span_id`, so a case's model calls,
tool calls and steps form a tree you can rebuild from the JSONL file alone. The trace id
is derived from the workflow id, so a case keeps ONE trace even when it pauses for a
reviewer, the process restarts, and it resumes hours later.

Spans always go to `<data_dir>/traces/<workflow_id>.jsonl`. With MC_OTEL_EXPORTER set
(and the `otel` extra installed) they are ALSO exported as real OpenTelemetry spans
under the same trace id:

  - `console`: printed by the SDK's ConsoleSpanExporter (handy to see it work);
  - `otlp`:    sent over OTLP/HTTP, configured by the standard OTEL_EXPORTER_OTLP_* vars
               (Jaeger, Grafana Tempo, Honeycomb and Langfuse all accept it).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import secrets
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar("mc_current_span", default=None)
_otel_tracer = None  # set by configure_otel when an exporter is configured


def reset_otel() -> None:
    """Test hook: stop exporting."""
    global _otel_tracer
    _otel_tracer = None


def configure_otel(exporter: str) -> str:
    """Install a real OpenTelemetry tracer provider. Returns what was configured.

    Idempotent per process. Unknown values and a missing `otel` extra are reported in
    the return value (and the health endpoint) rather than raising: tracing must never
    stop the platform from starting.
    """
    global _otel_tracer
    if not exporter:
        return "jsonl"
    if _otel_tracer is not None:
        return exporter
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    except ImportError:
        logger.warning("MC_OTEL_EXPORTER=%s but the `otel` extra is not installed", exporter)
        return "jsonl (otel extra not installed)"
    provider = TracerProvider(resource=Resource.create({"service.name": "mission-control"}))
    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            logger.warning("MC_OTEL_EXPORTER=otlp needs opentelemetry-exporter-otlp-proto-http")
            return "jsonl (otlp exporter not installed)"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        logger.warning("unknown MC_OTEL_EXPORTER=%r; spans stay in JSONL only", exporter)
        return "jsonl"
    # Our own provider, not the process-global one: an embedding app keeps its own
    # OpenTelemetry setup, and tests can reset this without fighting a set-once global.
    _otel_tracer = provider.get_tracer("mission-control")
    return exporter


def trace_id_for(workflow_id: str) -> str:
    """32 hex chars, stable for a workflow, valid as an OpenTelemetry trace id."""
    return hashlib.sha256(workflow_id.encode()).hexdigest()[:32]


class Tracer:
    """A span recorder scoped to one workflow (one trace)."""

    def __init__(self, workflow_id: str, data_dir: Path) -> None:
        self.workflow_id = workflow_id
        self.trace_id = trace_id_for(workflow_id)
        self.spans: list[dict[str, Any]] = []
        sink = data_dir / "traces"
        sink.mkdir(parents=True, exist_ok=True)
        self._path = sink / f"{workflow_id}.jsonl"

    @contextmanager
    def span(self, name: str, **attrs: Any):
        span_id = secrets.token_hex(8)
        parent = _current_span.get()
        token = _current_span.set(span_id)
        start = time.time()
        record: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": span_id,
            "parent_span_id": parent,
            "workflow_id": self.workflow_id,
            "name": name,
            "attrs": attrs,
            "ts": start,
        }
        otel_cm = self._otel_span(name, parent, attrs)
        try:
            if otel_cm is not None:
                otel_cm.__enter__()
            yield record
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - record, then re-raise
            record["status"] = "error"
            record["error"] = str(exc)[:300]
            raise
        finally:
            record["duration_ms"] = round((time.time() - start) * 1000, 2)
            _current_span.reset(token)
            self.spans.append(record)
            self._append(record)
            if otel_cm is not None:
                otel_cm.__exit__(None, None, None)

    def _otel_span(self, name: str, parent: str | None, attrs: dict[str, Any]):
        """A real OTel span in this workflow's trace, or None when OTel is off."""
        if _otel_tracer is None:
            return None
        from opentelemetry import trace
        from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

        context = None
        if parent is None:
            # Top-level span: parent it on a remote context carrying the workflow's
            # trace id, so every step of the case (before and after a pause) lands
            # in the same trace.
            remote = SpanContext(
                trace_id=int(self.trace_id, 16),
                span_id=int(secrets.token_hex(8), 16),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            context = trace.set_span_in_context(NonRecordingSpan(remote))
        clean = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in attrs.items()}
        clean["mc.workflow_id"] = self.workflow_id
        return _otel_tracer.start_as_current_span(name, context=context, attributes=clean)

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:  # pragma: no cover - tracing must never break the run
            pass
