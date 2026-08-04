"""Spans, costs and the running total.

The trace is the product of this project, not a debugging aid bolted on at the
end. Every span carries the agent id and the prompt version that produced it,
because when the answer is wrong those are the first two things you need and
the two most commonly missing.

A span can be green and still be part of a wrong answer. See the `warnings`
field: it is how a span says "I succeeded, and you should not trust me".
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .config import price_for

SpanStatus = Literal["running", "ok", "error", "timeout", "killed", "skipped"]


def _now() -> float:
    return time.perf_counter()


@dataclass
class Span:
    name: str
    stage: int
    agent_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    kind: Literal["stage", "model", "tool", "retrieval", "saga"] = "stage"

    prompt_version: str | None = None
    model: str | None = None

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0

    status: SpanStatus = "running"
    started_at: float = field(default_factory=_now)
    ended_at: float | None = None

    # Budget state at the moment this span started. Visible in the waterfall so
    # a reader can watch the remainder shrink as it is passed down.
    deadline_remaining_s: float | None = None
    tokens_remaining: int | None = None
    timeout_s: float | None = None

    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    # Semantic warnings. These do NOT turn the span red, deliberately. A stale
    # or corrupted passage produces a perfectly healthy span.
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else _now()
        return (end - self.started_at) * 1000.0

    @property
    def cost_usd(self) -> float:
        if not self.model:
            return 0.0
        return price_for(self.model).cost(
            self.input_tokens, self.cached_tokens, self.output_tokens
        )

    def finish(self, status: SpanStatus = "ok", error: str | None = None) -> "Span":
        self.ended_at = _now()
        self.status = status
        self.error = error
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "stage": self.stage,
            "agent_id": self.agent_id,
            "kind": self.kind,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "status": self.status,
            "duration_ms": round(self.duration_ms, 1),
            "deadline_remaining_s": (
                round(self.deadline_remaining_s, 2)
                if self.deadline_remaining_s is not None
                else None
            ),
            "tokens_remaining": self.tokens_remaining,
            "timeout_s": self.timeout_s,
            "detail": self.detail,
            "error": self.error,
            "warnings": self.warnings,
        }


class Trace:
    """Collects spans for one request and knows the running total."""

    def __init__(self, request_id: str, tenant_id: str, mode: str) -> None:
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.mode = mode
        self.spans: list[Span] = []
        self._listeners: list[Any] = []

    def start(
        self,
        name: str,
        stage: int,
        agent_id: str,
        *,
        parent: Span | None = None,
        kind: str = "stage",
        prompt_key: str | None = None,
        **kwargs: Any,
    ) -> Span:
        from .config import PROMPT_VERSIONS

        span = Span(
            name=name,
            stage=stage,
            agent_id=agent_id,
            parent_id=parent.span_id if parent else None,
            kind=kind,  # type: ignore[arg-type]
            prompt_version=PROMPT_VERSIONS.get(prompt_key or "") if prompt_key else None,
            **kwargs,
        )
        self.spans.append(span)
        return span

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    @property
    def cached_tokens(self) -> int:
        return sum(s.cached_tokens for s in self.spans)

    @property
    def model_calls(self) -> int:
        return sum(1 for s in self.spans if s.kind == "model")

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.spans if s.kind in ("tool", "retrieval"))

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "mode": self.mode,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "spans_green": sum(1 for s in self.spans if s.status == "ok"),
            "spans_total": len(self.spans),
            "semantic_warnings": sum(len(s.warnings) for s in self.spans),
        }

    def to_dict(self) -> dict[str, Any]:
        # Offsets are relative to the first span, so the waterfall can place a
        # bar rather than just size it. Without this, three parallel branches
        # and three sequential ones draw identically.
        t0 = min((s.started_at for s in self.spans), default=0.0)
        out = []
        for s in self.spans:
            d = s.to_dict()
            d["started_offset_ms"] = round((s.started_at - t0) * 1000.0, 1)
            out.append(d)
        return {"summary": self.summary(), "spans": out}
