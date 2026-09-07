"""Phase III - Deployment: the judge in two roles at once.

    generate -> judge -> revise -> judge -> ... -> serve or DROP

The same tuned judge plays both parts, and the fact that it is the same one is
the point:

**Gate.** It rejects artefacts failing any must-have criterion. Soft failures
are recorded and served.

**Critic.** Its rejection reason is appended to the generator's prompt as the
instruction for the next attempt. This is what makes revision better than
resampling, and it is also why a right-verdict-but-wrong-reason rejection is a
real defect rather than a philosophical one: the wrong reason steers the next
draft at the wrong problem, so the retry budget is spent and the artefact is
dropped anyway.

The asymmetry, which is the whole design
----------------------------------------
When the retry budget is exhausted, the artefact is DROPPED. Not served with a
warning, not served as the best of a bad set - dropped.

    a bad artefact served  -> reaches a user, damages trust, cannot be recalled
    a good artefact dropped -> one missed opportunity, costing nothing

Those are not the same error, and a system that treats them as the same will
optimise for coverage and pay for it in trust. This is the single most
important line in the module and it is one `if`.

``on_budget_exhausted`` in configs/base.yaml can switch it to ``serve_best`` or
``serve_flagged``, and both are there so you can measure what the choice costs
rather than take the claim on faith. They should not be your default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .domain import Domain, Record
from .generator import Generator
from .judge import JudgePanel, Verdict
from .runtime import Runtime

log = logging.getLogger(__name__)

SERVED_CLEAN = "served_without_revision"
SERVED_REVISED = "served_after_revision"
DROPPED = "dropped"


@dataclass
class Attempt:
    index: int
    artefact: str
    gate_failures: list[dict[str, str]] = field(default_factory=list)
    soft_failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed_gate(self) -> bool:
        return not self.gate_failures


@dataclass
class ServedResult:
    record_id: str
    outcome: str
    artefact: str | None
    attempts: list[Attempt] = field(default_factory=list)
    soft_failures: list[dict[str, str]] = field(default_factory=list)
    usd: float = 0.0
    llm_calls: int = 0

    @property
    def revisions(self) -> int:
        """Number of REVISIONS, which is attempts minus the first draft.

        Off-by-one here quietly corrupts the retry-budget curve: k=0 has to mean
        "first generation, no revision" for the curve to be readable, and a
        version that counts attempts instead shifts every point one place left
        and makes K=3 look like it buys more than it does.
        """
        return max(0, len(self.attempts) - 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "outcome": self.outcome,
            "artefact": self.artefact,
            "revisions": self.revisions,
            "soft_failures": self.soft_failures,
            "usd": round(self.usd, 6),
            "llm_calls": self.llm_calls,
            "trace": [
                {
                    "attempt": a.index,
                    "artefact": a.artefact,
                    "gate_failures": a.gate_failures,
                    "soft_failures": a.soft_failures,
                }
                for a in self.attempts
            ],
        }


class ServingLoop:
    def __init__(
        self,
        runtime: Runtime,
        domain: Domain,
        rubrics: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.domain = domain
        self.config = config or {}
        self.generator = Generator(runtime, domain)
        self.panel = JudgePanel(runtime, domain, rubrics)
        self.max_retries = int(self.config.get("max_retries", 3))
        self.reason_guided = bool(self.config.get("reason_guided_revision", True))
        self.on_exhausted = self.config.get("on_budget_exhausted", "drop")

    def serve(self, record: Record, *, max_retries: int | None = None) -> ServedResult:
        budget = self.max_retries if max_retries is None else max_retries
        start_usd = self.runtime.usage.total_usd
        start_calls = self.runtime.usage.total_calls

        attempts: list[Attempt] = []
        critique: str | None = None

        for index in range(budget + 1):  # +1: the first draft is not a retry
            artefact = self.generator.write(record, critique=critique, attempt=index)
            verdicts = self.panel.judge_all(record, artefact)
            gate = self.panel.gate_failures(verdicts)
            soft = self.panel.soft_failures(verdicts)

            attempt = Attempt(
                index=index,
                artefact=artefact,
                gate_failures=[_as_row(v) for v in gate],
                soft_failures=[_as_row(v) for v in soft],
            )
            attempts.append(attempt)

            if attempt.passed_gate:
                return self._finish(
                    record, SERVED_CLEAN if index == 0 else SERVED_REVISED,
                    artefact, attempts, [_as_row(v) for v in soft],
                    start_usd, start_calls,
                )

            critique = self._critique(gate) if self.reason_guided else None

        return self._exhausted(record, attempts, start_usd, start_calls)

    def _critique(self, gate_failures: list[Verdict]) -> str:
        """Turn rejections into a revision instruction.

        The failure mode is included alongside the prose because the prose is a
        model's sentence and the mode is a stable token. A generator that reads
        "generic_filler" knows exactly what to change; one reading only "this
        could describe anything" may reasonably rewrite the whole line and lose
        the parts that passed.
        """
        return "\n".join(
            f"- [{v.criterion_id}/{v.failure_mode or 'unspecified'}] {v.reason}"
            for v in gate_failures
        )

    def _exhausted(
        self,
        record: Record,
        attempts: list[Attempt],
        start_usd: float,
        start_calls: int,
    ) -> ServedResult:
        if self.on_exhausted == "drop":
            log.info(
                "dropping %s after %d attempts; last failures: %s",
                record.id, len(attempts),
                [f["criterion"] for f in attempts[-1].gate_failures],
            )
            return self._finish(record, DROPPED, None, attempts, [], start_usd, start_calls)

        # Both non-default paths serve something that failed the gate. They
        # exist to be measured against `drop`, not to be switched on because the
        # coverage number looks better - which it will, right up until the first
        # complaint about an explanation that gave away an ending.
        last = attempts[-1]
        log.warning(
            "serving %s despite %d gate failure(s) because on_budget_exhausted=%r. "
            "This ships artefacts the gate rejected.",
            record.id, len(last.gate_failures), self.on_exhausted,
        )
        return self._finish(
            record, SERVED_REVISED, last.artefact, attempts,
            last.gate_failures + last.soft_failures, start_usd, start_calls,
        )

    def _finish(
        self,
        record: Record,
        outcome: str,
        artefact: str | None,
        attempts: list[Attempt],
        soft: list[dict[str, str]],
        start_usd: float,
        start_calls: int,
    ) -> ServedResult:
        return ServedResult(
            record_id=record.id,
            outcome=outcome,
            artefact=artefact,
            attempts=attempts,
            soft_failures=soft,
            usd=self.runtime.usage.total_usd - start_usd,
            llm_calls=self.runtime.usage.total_calls - start_calls,
        )


def retry_curve(results: list[ServedResult], max_k: int) -> list[dict[str, Any]]:
    """Cumulative pass rate against revision budget k. The paper's Figure 3.

    Read it before choosing ``max_retries``, because it is the only thing that
    turns K from a hyperparameter into a decision. Three things it tells you:

    * **Where it flattens.** Gains are monotone in k and diminish fast. If most
      of the achievable lift is captured by k=3, every retry past that is a
      linear cost increase on every single request for a rounding error of
      quality.

    * **Whether your generator is worth revising.** A weak generator stays low
      at every k. Revision AMPLIFIES a capable generator; it does not substitute
      for one, and a flat low curve is a signal to fix the writer rather than to
      buy more retries.

    * **Which component regressed.** The k=0 point is the generator's unaided
      pass rate. A sustained drop there is a generator-side regression. A drop
      in the later points with k=0 unchanged is the judge or the critique path.
      Without the curve, both look identical from the aggregate pass rate, and
      you will debug the wrong one.
    """
    total = len(results)
    if total == 0:
        return []
    curve: list[dict[str, Any]] = []
    for k in range(max_k + 1):
        passed = sum(
            1
            for r in results
            if r.outcome in (SERVED_CLEAN, SERVED_REVISED) and r.revisions <= k
        )
        curve.append(
            {
                "k": k,
                "cumulative_pass_rate": round(passed / total, 4),
                "passed": passed,
                "total": total,
            }
        )
    return curve


def _as_row(verdict: Verdict) -> dict[str, str]:
    return {
        "criterion": verdict.criterion_id,
        "failure_mode": verdict.failure_mode or "",
        "reason": verdict.reason,
    }
