"""Score one rubric over a set of labelled examples.

The inner loop of the whole project. RART calls it once per split per iteration,
the final report calls it once on the held-out test set, and the drift monitor
calls it on each week's freshly-rated sample. One implementation, so those three
numbers are comparable - three separate scoring paths that drifted apart would
be a quiet disaster, because nothing would look wrong.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .domain import Criterion, Domain, LabelledExample
from .judge import Judge
from .meta_judge import MetaJudge
from .metrics import Judgement, Metrics, compute, confusion
from .runtime import Runtime

log = logging.getLogger(__name__)


def score(
    runtime: Runtime,
    domain: Domain,
    criterion: Criterion,
    rubric: str,
    examples: Iterable[LabelledExample],
    *,
    weights: dict[str, float] | None = None,
    assess_reasoning: bool = True,
) -> tuple[list[Judgement], Metrics]:
    """Judge every example, then meta-judge only the agreed-fails.

    ``assess_reasoning=False`` reproduces the paper's *vanilla* ablation: label
    signal only, no reasoning meta-judge, no reason mismatches in the focus set.
    Running both arms is the only way to find out whether reasoning alignment is
    buying you anything on YOUR criteria - and on a criterion whose seed rubric
    is already near ceiling, the honest answer is that it is not.
    """
    judge = Judge(runtime, domain, criterion, rubric)
    meta = MetaJudge(runtime) if assess_reasoning else None

    judgements: list[Judgement] = []
    for example in examples:
        human_label = example.labels[criterion.id]
        record = domain.record(example.record_id)
        verdict = judge.judge(record, example.artefact)

        judgement = Judgement(
            example_id=example.id,
            human_label=human_label,
            judge_label=verdict.label,
            human_rationale=example.rationales.get(criterion.id, ""),
            judge_reason=verdict.reason,
            human_mode=example.failure_modes.get(criterion.id),
            judge_mode=verdict.failure_mode,
        )

        # Only agreed-fails get a reasoning verdict. See src/meta_judge.py for
        # why that restriction is structural rather than a saving.
        if meta is not None and judgement.is_agreed_fail and judgement.human_rationale:
            judgement.rationale_agrees = meta.agrees(
                example.artefact,
                verdict.reason,
                verdict.failure_mode,
                judgement.human_rationale,
                judgement.human_mode,
            )

        judgements.append(judgement)

    return judgements, compute(judgements, weights)


def report(
    judgements: list[Judgement], metrics: Metrics, *, criterion_id: str = ""
) -> dict[str, Any]:
    return {
        "criterion": criterion_id,
        "metrics": metrics.as_dict(),
        "confusion": confusion(judgements),
        # The two lists a reviewer actually reads. False passes are the bad
        # artefacts that would have reached users; false fails are the good ones
        # that would have been regenerated or dropped. Aggregate metrics tell
        # you how many; only the ids tell you which, and "which" is what you
        # need to decide whether the judge is wrong or the labels are.
        "false_passes": [
            j.example_id
            for j in judgements
            if j.human_label == "FAIL" and j.judge_label == "PASS"
        ],
        "false_fails": [
            j.example_id
            for j in judgements
            if j.human_label == "PASS" and j.judge_label == "FAIL"
        ],
        "reason_mismatches": [
            j.example_id for j in judgements if j.rationale_agrees is False
        ],
    }


def focus_set(judgements: list[Judgement], examples: dict[str, LabelledExample]) -> list[dict[str, Any]]:
    """Algorithm 1, lines 11-14: what the reflector is allowed to see.

        X  = label mismatches            (the judge got the verdict wrong)
        N_c = agreed-fails, wrong reason (right verdict, wrong reason)
        focus = X union N_c

    Agreed-fails where the reasoning also agrees are excluded on purpose. Those
    are what the rubric already gets right, and handing them to the reflector
    invites it to "improve" working clauses - which is how a tuning run makes
    things worse while every iteration looks busy.
    """
    focus: list[dict[str, Any]] = []
    for judgement in judgements:
        wrong_label = not judgement.label_matches
        wrong_reason = judgement.is_agreed_fail and judgement.rationale_agrees is False
        if not (wrong_label or wrong_reason):
            continue
        example = examples.get(judgement.example_id)
        focus.append(
            {
                "example_id": judgement.example_id,
                "artefact": example.artefact if example else "",
                "human_label": judgement.human_label,
                "judge_label": judgement.judge_label,
                "human_rationale": judgement.human_rationale,
                "judge_reason": judgement.judge_reason,
                "error_type": "label_mismatch" if wrong_label else "reason_mismatch",
            }
        )
    return focus
