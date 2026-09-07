"""Phase II service: run RART, and stage the result behind a human."""

from __future__ import annotations

from typing import Any

from ..benchmark import Benchmark
from ..evaluate import report, score
from ..rart import tune
from ._context import Context, build_context


class TuningService:
    def __init__(self, context: Context | None = None) -> None:
        self.ctx = context or build_context()
        self.benchmark = Benchmark(self.ctx.domain, self.ctx.config.benchmark)

    def tune(
        self,
        criterion_id: str,
        *,
        reasoning_alignment: bool | None = None,
        stage: bool = True,
    ) -> dict[str, Any]:
        criterion = self.ctx.domain.criterion(criterion_id)
        splits = self.benchmark.split_for(criterion_id)

        config = dict(self.ctx.config.rart)
        if reasoning_alignment is not None:
            config["reasoning_alignment"] = reasoning_alignment

        result = tune(
            self.ctx.runtime,
            self.ctx.domain,
            criterion,
            splits.train,
            splits.validation,
            config,
        )

        payload = result.as_dict()
        if stage:
            # STAGED, never live. A tuned rubric changes what the gate rejects,
            # which changes what reaches users. Promoting it is
            # `run.py promote`, and it is a separate command precisely so that
            # somebody has to read the diff first. An auto-deploying tuner is a
            # system rewriting its own success criteria unsupervised, and it
            # will eventually conclude that it is doing well.
            path = self.ctx.save_rubric(criterion_id, result.best_rubric, staged=True)
            payload["staged_at"] = str(path)
            payload["promote_with"] = f"python run.py promote --criterion {criterion_id}"

        return payload

    def ablation(self, criterion_id: str) -> dict[str, Any]:
        """RART versus vanilla: same splits, same scoring, only the SIGNAL differs.

        The comparison the paper runs, and the one worth running on your own
        criteria before deciding reasoning alignment earns its extra calls.

        The subtle part is the second half of this method, and getting it wrong
        produces a result that is flattering and meaningless.

        Vanilla TUNES without the reasoning meta-judge, so during its own loop
        reasoning agreement is never computed and enters its weighted score as
        zero. Comparing the two arms' internal scores therefore hands RART a
        free point on a term the other arm did not measure - a gap that would
        appear even if the two produced byte-identical rubrics. It is a
        difference in instrumentation reported as a difference in quality.

        So both tuned rubrics are RE-SCORED here on the same validation split
        with reasoning assessment ON. Only then are the numbers comparable, and
        only the tuning signal differs, which is the thing the ablation is
        supposed to isolate.
        """
        aligned = self.tune(criterion_id, reasoning_alignment=True, stage=False)
        vanilla = self.tune(criterion_id, reasoning_alignment=False, stage=False)

        criterion = self.ctx.domain.criterion(criterion_id)
        splits = self.benchmark.split_for(criterion_id)
        weights = self.ctx.config.rart.get("weights")

        scored = {}
        for name, arm in (("rart", aligned), ("vanilla", vanilla)):
            judgements, metrics = score(
                self.ctx.runtime,
                self.ctx.domain,
                criterion,
                arm["best_rubric"],
                splits.validation,
                weights=weights,
                assess_reasoning=True,  # identical instrumentation for both arms
            )
            scored[name] = report(judgements, metrics, criterion_id=criterion_id)

        delta = (
            scored["rart"]["metrics"]["weighted"]
            - scored["vanilla"]["metrics"]["weighted"]
        )
        return {
            "criterion": criterion_id,
            "comparable_scores": scored,
            "delta_weighted": round(delta, 4),
            "identical_rubrics": aligned["best_rubric"] == vanilla["best_rubric"],
            "tuning_traces": {"rart": aligned, "vanilla": vanilla},
            "caveat": (
                "One seed, one split. The paper reports eight seeds with the "
                "split reshuffled each time, and on a benchmark this size a "
                "single split is not evidence of anything - re-run across "
                "several values of benchmark.seed before drawing a conclusion. "
                "If identical_rubrics is true, the two arms converged on the "
                "same text and any delta here is scoring noise, not an effect."
            ),
        }

    def promote(self, criterion_id: str) -> dict[str, Any]:
        staged = self.ctx.rubric_path(criterion_id, staged=True)
        if not staged.exists():
            raise FileNotFoundError(
                f"nothing staged for {criterion_id!r}. Run "
                f"`python run.py tune --criterion {criterion_id}` first."
            )
        rubric = staged.read_text(encoding="utf-8")
        live = self.ctx.rubric_path(criterion_id)
        previous = live.read_text(encoding="utf-8") if live.exists() else None

        if previous is not None:
            # Keep the outgoing rubric. Rollback has to be a file copy at three
            # in the morning, not a re-run of a tuning loop whose inputs have
            # since changed and which will not reproduce the rubric you had.
            self.ctx.save_rubric(f"{criterion_id}.previous", previous, staged=False)

        self.ctx.save_rubric(criterion_id, rubric, staged=False)
        return {
            "criterion": criterion_id,
            "promoted": str(live),
            "rollback_available": previous is not None,
        }
