"""Alignment metrics: specificity, recall, reasoning agreement, weighted score.

Three numbers, and the third is the one most evaluation harnesses do not have.

    Specificity  (fail-recall)  of the artefacts a human failed, what fraction
                                did the judge also fail?
    Recall       (pass-recall)  of the artefacts a human passed, what fraction
                                did the judge also pass?
    RA_neg       (reasoning)    of the artefacts a human failed, what fraction
                                did the judge fail FOR THE SAME REASON?

Note the denominator on the third. It is every human failure, not every case
where the judge happened to agree. Dividing by agreed-fails instead would make
a judge that catches two things and explains both perfectly outscore one that
catches ten and explains nine - which is backwards, and is the natural mistake
to make when implementing Eq. 3 from the paper.

Why not accuracy or F1
----------------------
Accuracy on a class-balanced benchmark is a coin-flip baseline of 0.50 with no
interpretation. Worse, it hides the asymmetry that decides the whole system: a
bad artefact the judge passes reaches a customer and cannot be recalled, while
a good one the judge fails costs a regeneration. Those are not the same error
and no single blended number should let them cancel out. The weighted score in
:func:`weighted_score` keeps them apart and prices them explicitly - specificity
at 3, the others at 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Judgement:
    """One judge verdict on one example, paired with the human's."""

    example_id: str
    human_label: str
    judge_label: str
    human_rationale: str = ""
    judge_reason: str = ""
    human_mode: str | None = None
    judge_mode: str | None = None
    #: Filled in by the meta-judge, and only for agreed-fails. None means
    #: "not assessed", which is different from "disagreed" and must not be
    #: counted as either.
    rationale_agrees: bool | None = None

    @property
    def label_matches(self) -> bool:
        return self.human_label == self.judge_label

    @property
    def is_agreed_fail(self) -> bool:
        return self.human_label == FAIL and self.judge_label == FAIL


@dataclass
class Metrics:
    specificity: float | None
    recall: float | None
    reasoning_agreement: float | None
    n_human_fail: int
    n_human_pass: int
    n_agreed_fail: int
    n_assessed_rationales: int
    weighted: float = 0.0
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "specificity": _round(self.specificity),
            "recall": _round(self.recall),
            "reasoning_agreement": _round(self.reasoning_agreement),
            "weighted": round(self.weighted, 4),
            "n_human_fail": self.n_human_fail,
            "n_human_pass": self.n_human_pass,
            "n_agreed_fail": self.n_agreed_fail,
            "n_assessed_rationales": self.n_assessed_rationales,
            "ci95": {
                k: [round(lo, 3), round(hi, 3)] for k, (lo, hi) in self.intervals.items()
            },
        }

    def clears(self, targets: dict[str, float]) -> bool:
        """Have all three metrics reached their per-criterion floor?

        An unmeasurable metric (None, because its denominator was empty) does
        NOT clear. Treating "no data" as success is how an early-stopping rule
        ends a tuning run at iteration 0 and reports it as a triumph.
        """
        for name, floor in targets.items():
            value = getattr(self, name, None)
            if value is None or value < floor:
                return False
        return True


def compute(
    judgements: Iterable[Judgement], weights: dict[str, float] | None = None
) -> Metrics:
    rows = list(judgements)
    weights = weights or {"specificity": 3.0, "recall": 1.0, "reasoning_agreement": 1.0}

    human_fail = [r for r in rows if r.human_label == FAIL]
    human_pass = [r for r in rows if r.human_label == PASS]
    agreed_fail = [r for r in rows if r.is_agreed_fail]

    specificity = _ratio(sum(1 for r in human_fail if r.judge_label == FAIL), len(human_fail))
    recall = _ratio(sum(1 for r in human_pass if r.judge_label == PASS), len(human_pass))

    assessed = [r for r in agreed_fail if r.rationale_agrees is not None]
    reasoning = _ratio(sum(1 for r in assessed if r.rationale_agrees), len(human_fail))

    metrics = Metrics(
        specificity=specificity,
        recall=recall,
        reasoning_agreement=reasoning,
        n_human_fail=len(human_fail),
        n_human_pass=len(human_pass),
        n_agreed_fail=len(agreed_fail),
        n_assessed_rationales=len(assessed),
    )
    metrics.weighted = weighted_score(metrics, weights)
    metrics.intervals = {
        "specificity": wilson(
            sum(1 for r in human_fail if r.judge_label == FAIL), len(human_fail)
        ),
        "recall": wilson(
            sum(1 for r in human_pass if r.judge_label == PASS), len(human_pass)
        ),
        "reasoning_agreement": wilson(
            sum(1 for r in assessed if r.rationale_agrees), len(human_fail)
        ),
    }
    return metrics


def weighted_score(metrics: Metrics, weights: dict[str, float]) -> float:
    """The single number RART optimises.

    Unmeasurable components contribute zero rather than being skipped. Skipping
    them would let a rubric that produces no agreed-fails - and therefore no
    measurable reasoning agreement - score identically to one that produces
    many and explains them all correctly. The optimiser would then have no
    reason to prefer the second.
    """
    total = 0.0
    for name, weight in weights.items():
        value = getattr(metrics, name, None)
        total += weight * (value or 0.0)
    return total


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval.

    Printed beside every metric because this benchmark is small, and a point
    estimate over eight examples invites a conclusion it cannot support. On a
    six-example test split the interval for 5/6 runs roughly 0.42 to 0.99 -
    which is the honest way of saying that a five-point difference between two
    rubrics means nothing at all.

    Wilson rather than the textbook normal approximation because the normal one
    is badly wrong at exactly the sample sizes and extreme proportions this
    project operates at: at 8/8 it produces the interval [1.0, 1.0], asserting
    certainty from eight observations.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def confusion(judgements: Iterable[Judgement]) -> dict[str, int]:
    rows = list(judgements)
    return {
        "agreed_pass": sum(1 for r in rows if r.human_label == PASS and r.judge_label == PASS),
        "agreed_fail": sum(1 for r in rows if r.is_agreed_fail),
        # The dangerous one: a human said this is bad and the judge let it
        # through. In production these are the artefacts that reach users.
        "false_pass": sum(1 for r in rows if r.human_label == FAIL and r.judge_label == PASS),
        # The merely expensive one: rejected something good, so it gets
        # regenerated or dropped.
        "false_fail": sum(1 for r in rows if r.human_label == PASS and r.judge_label == FAIL),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
