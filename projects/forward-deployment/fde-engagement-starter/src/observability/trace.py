"""Per-request cost and latency tracing.

This one is GIVEN TO YOU, working, on purpose.

Not because it is unimportant, but because the failure mode it prevents happens
in week one and a learner who has to build it first will skip it. Use it from the
start, and when the customer asks "what is this going to cost us at 10x volume"
you answer with a measurement instead of an estimate.

The design is deliberately small: a context manager that records one span. There
is no sampling, no exporter, no vendor. Adding OpenTelemetry later is a contained
change, and doing it now would be the kind of premature platform-building that
makes an engagement late.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Illustrative per-million-token prices so the arithmetic is visible. These are
# NOT current prices and are not a claim about any vendor. Set them from the
# provider's own pricing page for whatever model you actually use, and re-check
# before you put a number in front of a customer.
PRICE_PER_MTOK_INPUT = 0.10
PRICE_PER_MTOK_OUTPUT = 0.40


@dataclass
class Span:
    """One traced operation."""

    span_id: str
    name: str
    started_at: float
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tenant: str = "northwind"
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects spans in memory and can dump them as JSONL.

    In-memory is the right call for a walking skeleton. The moment you need this
    across processes, write it to the same place as the audit log rather than
    adding a dependency.
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, tenant: str = "northwind", **metadata: Any) -> Iterator[Span]:
        s = Span(
            span_id=uuid.uuid4().hex[:12],
            name=name,
            started_at=time.time(),
            tenant=tenant,
            metadata=dict(metadata),
        )
        start = time.perf_counter()
        try:
            yield s
        finally:
            s.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            s.cost_usd = estimate_cost(s.input_tokens, s.output_tokens)
            self.spans.append(s)

    def total_cost(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    def p95_latency_ms(self) -> float:
        """p95, not mean.

        Mean latency hides the tail, and the tail is what a user notices and what
        an SLA is written against.
        """
        if not self.spans:
            return 0.0
        ordered = sorted(s.duration_ms for s in self.spans)
        idx = min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[idx]

    def summary(self) -> dict[str, Any]:
        return {
            "spans": len(self.spans),
            "total_cost_usd": self.total_cost(),
            "p95_latency_ms": self.p95_latency_ms(),
            "by_name": {
                name: sum(1 for s in self.spans if s.name == name)
                for name in sorted({s.name for s in self.spans})
            },
        }

    def dump(self, path: Path) -> None:
        with Path(path).open("a", encoding="utf-8") as fh:
            for s in self.spans:
                fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Cost for one call at the prices above.

    Per call, tagged with a tenant. Aggregate cost across a deployment tells you
    nothing actionable; cost per tenant tells you which customer is about to
    surprise you.
    """
    return round(
        (input_tokens / 1_000_000) * PRICE_PER_MTOK_INPUT
        + (output_tokens / 1_000_000) * PRICE_PER_MTOK_OUTPUT,
        6,
    )


tracer = Tracer()
