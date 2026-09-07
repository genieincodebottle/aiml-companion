"""Phase II - Reasoning-Aligned Rubric Tuning.

No gradients. No fine-tuning. No weights are touched anywhere in this file.

**The rubric text is the parameter, and a reflector LLM is the optimiser.** The
loop is the shape of gradient descent with every numeric part replaced by
language: score the current rubric on the training split, collect the errors,
ask a model to propose a better rubric given those errors, keep it if validation
improves. That is the whole method, and it is why this runs on a laptop against
a hosted API instead of on a cluster.

What makes it *reasoning-aligned* is the focus set. A label-only loop sees
mistaken verdicts. This one also sees cases where the verdict was right and the
stated reason was not - see :mod:`src.meta_judge` for why that is a defect worth
paying for rather than a nicety.

Algorithm 1, faithfully
-----------------------
    R* <- R_0 ; s* <- -inf
    for t in 0 .. N-1:
        judgements <- score(D_train, J(R_t))
        s <- weighted metrics on D_val                     <- validation, not train
        if s > s*:  R*, s* <- R_t, s                       <- keep the best
        if all metrics clear their floors: break
        X   <- label mismatches
        N_c <- agreed-fails whose reasons mismatch
        R_{t+1} <- Reflect(R_t, X union N_c)
    return R*

Three details that are easy to get wrong, and each is load-bearing:

1. **Early stopping is on VALIDATION, never on train.** The reflector has seen
   every training error by construction, so training score rises almost
   monotonically whether or not the rubric got better. Selecting on it selects
   for memorisation.

2. **The best checkpoint is returned, not the last.** Rubric edits are not
   monotone improvements; iteration 4 is regularly worse than iteration 2. Any
   loop that returns R_N is reporting wherever it happened to stop.

3. **The test split is never touched here.** It is scored once, at the end, by
   ``run.py eval``. The moment tuning consults it, the final number is a fit
   rather than a result, and nothing in the output would reveal that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .domain import Criterion, Domain, LabelledExample
from .evaluate import focus_set, report, score
from .metrics import Metrics
from .prompts import REFLECTOR_SYSTEM, reflector_prompt
from .runtime import Runtime

log = logging.getLogger(__name__)


@dataclass
class Iteration:
    index: int
    rubric: str
    train: dict[str, Any]
    validation: dict[str, Any]
    weighted: float
    focus_size: int
    label_mismatches: int
    reason_mismatches: int
    accepted: bool = False


@dataclass
class TuningResult:
    criterion_id: str
    seed_rubric: str
    best_rubric: str
    best_score: float
    best_iteration: int
    iterations: list[Iteration] = field(default_factory=list)
    stopped_because: str = ""
    reasoning_alignment: bool = True

    @property
    def improved(self) -> bool:
        return self.best_iteration > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion_id,
            "reasoning_alignment": self.reasoning_alignment,
            "best_iteration": self.best_iteration,
            "best_weighted_score": round(self.best_score, 4),
            "stopped_because": self.stopped_because,
            "improved_on_seed": self.improved,
            "iterations": [
                {
                    "index": it.index,
                    "weighted": round(it.weighted, 4),
                    "accepted": it.accepted,
                    "validation": it.validation["metrics"],
                    "focus_size": it.focus_size,
                    "label_mismatches": it.label_mismatches,
                    "reason_mismatches": it.reason_mismatches,
                }
                for it in self.iterations
            ],
            "best_rubric": self.best_rubric,
        }


def tune(
    runtime: Runtime,
    domain: Domain,
    criterion: Criterion,
    train: list[LabelledExample],
    validation: list[LabelledExample],
    config: dict[str, Any],
) -> TuningResult:
    weights = dict(config.get("weights") or {})
    targets = dict(config.get("targets") or {})
    max_iterations = int(config.get("max_iterations", 6))

    # The ablation switch. False is the paper's "vanilla" arm: the reflector
    # sees only label mismatches, so a right-verdict/wrong-reason case never
    # reaches it. Run both and compare - see docs/results/ for what happens on
    # a criterion whose seed rubric is already near ceiling.
    aligned = bool(config.get("reasoning_alignment", True))

    by_id = {e.id: e for e in train}
    rubric = domain.seed_rubric(criterion.id)
    best_rubric, best_score, best_index = rubric, float("-inf"), 0
    iterations: list[Iteration] = []
    stopped = f"reached max_iterations={max_iterations}"

    for index in range(max_iterations):
        train_judgements, train_metrics = score(
            runtime, domain, criterion, rubric, train,
            weights=weights, assess_reasoning=aligned,
        )
        _, val_metrics = score(
            runtime, domain, criterion, rubric, validation,
            weights=weights, assess_reasoning=aligned,
        )

        focus = focus_set(train_judgements, by_id)
        iteration = Iteration(
            index=index,
            rubric=rubric,
            train=report(train_judgements, train_metrics, criterion_id=criterion.id),
            validation=report([], val_metrics, criterion_id=criterion.id),
            weighted=val_metrics.weighted,
            focus_size=len(focus),
            label_mismatches=sum(1 for f in focus if f["error_type"] == "label_mismatch"),
            reason_mismatches=sum(1 for f in focus if f["error_type"] == "reason_mismatch"),
        )

        # Strictly greater. On a tie the earlier rubric wins, because it is the
        # simpler one - the reflector only ever adds - and a tie is not evidence
        # for the more complicated rubric.
        if val_metrics.weighted > best_score:
            best_rubric, best_score, best_index = rubric, val_metrics.weighted, index
            iteration.accepted = True

        iterations.append(iteration)
        log.info(
            "criterion=%s iter=%d weighted=%.3f spec=%s rec=%s ra=%s focus=%d",
            criterion.id, index, val_metrics.weighted,
            _fmt(val_metrics.specificity), _fmt(val_metrics.recall),
            _fmt(val_metrics.reasoning_agreement), len(focus),
        )

        if _clears(val_metrics, targets):
            stopped = "every metric cleared its target on validation"
            break
        if not focus:
            # No errors left to learn from. Continuing would ask the reflector
            # to improve a rubric it has been shown no evidence against, and it
            # will oblige - producing edits driven by nothing.
            stopped = "focus set was empty: no errors left to learn from"
            break

        proposed = _reflect(runtime, criterion, rubric, focus)
        if proposed.strip() == rubric.strip():
            # The optimiser looked at the errors and proposed nothing. Running
            # the remaining iterations would re-score an unchanged rubric
            # against an unchanged split and print the same row N times - real
            # money, online, for a guaranteed no-op.
            #
            # Reported as its own stop reason rather than folded into
            # "max_iterations", because the two mean different things. Hitting
            # the iteration cap says the search was still moving when it ran
            # out of budget. This says the search had stopped, and the useful
            # follow-up is to ask why the reflector could not see a fix - not
            # to raise max_iterations.
            stopped = (
                "the reflector proposed no change; the search converged with "
                f"{len(focus)} error(s) still in the focus set"
            )
            break
        rubric = proposed

    return TuningResult(
        criterion_id=criterion.id,
        seed_rubric=domain.seed_rubric(criterion.id),
        best_rubric=best_rubric,
        best_score=best_score,
        best_iteration=best_index,
        iterations=iterations,
        stopped_because=stopped,
        reasoning_alignment=aligned,
    )


def _reflect(
    runtime: Runtime, criterion: Criterion, rubric: str, focus: list[dict[str, Any]]
) -> str:
    """One optimiser step: propose a revised rubric from the focus set.

    An empty or nonsensical proposal returns the CURRENT rubric unchanged rather
    than raising. A reflector that misbehaves at iteration 4 should cost you one
    wasted iteration, not the four completed ones - and returning the current
    rubric means the next scoring pass simply reproduces this one, which shows
    up as a flat line in the trace rather than as a mystery.
    """
    completion = runtime.call(
        "reflector",
        reflector_prompt(criterion, rubric, focus),
        system=REFLECTOR_SYSTEM,
    )
    proposed = (completion.text or "").strip()

    if len(proposed.split()) < 25:
        log.warning(
            "reflector returned %d words for criterion %r; keeping the current "
            "rubric. A rubric this short cannot carry a pass condition, a fail "
            "condition and a boundary case.",
            len(proposed.split()), criterion.id,
        )
        return rubric

    dropped = _dropped_tags(rubric, proposed)
    if dropped:
        # Machine-readable tags are checks. Dropping one silently disables it,
        # and the metrics would show the loss without ever pointing at the
        # cause. Rejecting the proposal is cheaper than debugging that.
        log.warning(
            "reflector proposal for %r dropped the tag(s) %s; keeping the "
            "current rubric.", criterion.id, dropped,
        )
        return rubric

    return proposed


def _dropped_tags(before: str, after: str) -> list[str]:
    from .rules import parse_rubric

    old = {r.kind for r in parse_rubric(before)}
    new = {r.kind for r in parse_rubric(after)}
    return sorted(old - new)


def _clears(metrics: Metrics, targets: dict[str, float]) -> bool:
    return metrics.clears(targets) if targets else False


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
