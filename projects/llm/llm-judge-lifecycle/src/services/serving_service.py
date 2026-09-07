"""Phase III service: the generate/judge/revise loop, and the retry curve."""

from __future__ import annotations

from typing import Any

from ..serving import ServingLoop, retry_curve
from ._context import Context, build_context


class ServingService:
    def __init__(self, context: Context | None = None) -> None:
        self.ctx = context or build_context()
        self.loop = ServingLoop(
            self.ctx.runtime,
            self.ctx.domain,
            self.ctx.load_rubrics(),
            self.ctx.config.serving,
        )

    def serve(self, record_id: str, *, max_retries: int | None = None) -> dict[str, Any]:
        record = self.ctx.domain.record(record_id)
        return self.loop.serve(record, max_retries=max_retries).as_dict()

    def serve_all(self, *, limit: int | None = None) -> dict[str, Any]:
        records = list(self.ctx.domain.records.values())[:limit]
        results = [self.loop.serve(record) for record in records]

        outcomes: dict[str, int] = {}
        for result in results:
            outcomes[result.outcome] = outcomes.get(result.outcome, 0) + 1

        served = sum(v for k, v in outcomes.items() if k.startswith("served"))
        return {
            "n": len(results),
            "outcomes": outcomes,
            "pass_rate": round(served / len(results), 4) if results else 0.0,
            "usd_total": round(sum(r.usd for r in results), 6),
            # The number that decides the retry budget. Cost per SERVED artefact,
            # not per attempt: dropped artefacts consumed the full budget and
            # produced nothing, and their cost has to land somewhere. Divide by
            # attempts instead and every increase in K looks cheaper than it is.
            "usd_per_served": round(
                sum(r.usd for r in results) / served, 6
            ) if served else None,
            "results": [r.as_dict() for r in results],
        }

    def retry_curve(self, *, max_k: int | None = None, limit: int | None = None) -> dict[str, Any]:
        """Pass rate against revision budget. Read before setting max_retries.

        Serving runs at the FULL k so one pass produces the whole curve; every
        result records how many revisions it actually needed, and the cumulative
        rate at each k is read back from that. Re-serving the corpus once per k
        would cost k times as much and, because the generator is sampled rather
        than deterministic, would measure a slightly different system at every
        point.
        """
        k = max_k if max_k is not None else int(self.ctx.config.serving.get("max_retries", 3))
        records = list(self.ctx.domain.records.values())[:limit]
        results = [self.loop.serve(record, max_retries=k) for record in records]
        curve = retry_curve(results, k)

        return {
            "max_k": k,
            "curve": curve,
            "reason_guided_revision": self.loop.reason_guided,
            "reading_guide": (
                "k=0 is the generator's unaided pass rate: a sustained drop "
                "there is a generator regression, not judge drift. Where the "
                "curve flattens is where extra retries stop buying quality and "
                "start being a linear cost on every request. A curve that is "
                "flat AND low means the generator is too weak for revision to "
                "rescue - fix the writer, do not raise K."
            ),
        }
