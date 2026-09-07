"""Phase I service: inspect the benchmark, and grow it near the boundary."""

from __future__ import annotations

import json
from typing import Any

from ..benchmark import Benchmark
from ..evaluate import report, score
from ..prompts import SYNTHESIS_SYSTEM, synthesis_prompt
from ._context import Context, build_context


class BenchmarkService:
    def __init__(self, context: Context | None = None) -> None:
        self.ctx = context or build_context()
        self.benchmark = Benchmark(self.ctx.domain, self.ctx.config.benchmark)

    def report(self) -> dict[str, Any]:
        return {
            "stats": self.ctx.domain.stats(),
            "splits": self.benchmark.report(),
        }

    def evaluate(
        self, criterion_id: str, *, split: str = "test", staged: bool = False
    ) -> dict[str, Any]:
        """Score the current rubric on one split.

        ``split="test"`` is the final number, and it should be run once, at the
        end, after tuning has settled. Running it between iterations and picking
        the best is tuning on test with extra steps, and the result stops being
        a held-out estimate the first time you do it - with nothing in the
        output to show that it happened.
        """
        criterion = self.ctx.domain.criterion(criterion_id)
        splits = self.benchmark.split_for(criterion_id)
        examples = {"train": splits.train, "validation": splits.validation, "test": splits.test}[split]

        rubrics = self.ctx.load_rubrics()
        if staged:
            path = self.ctx.rubric_path(criterion_id, staged=True)
            if path.exists():
                rubrics[criterion_id] = path.read_text(encoding="utf-8")

        judgements, metrics = score(
            self.ctx.runtime,
            self.ctx.domain,
            criterion,
            rubrics[criterion_id],
            examples,
            weights=self.ctx.config.rart.get("weights"),
        )
        out = report(judgements, metrics, criterion_id=criterion_id)
        out["split"] = split
        out["n"] = len(examples)
        return out

    def synthesise(self, criterion_id: str, *, per_criterion: int | None = None) -> dict[str, Any]:
        """Phase I source (ii): boundary cases written by a model, labelled by a human.

        Rows are written with ``labels: {}`` and are NOT loaded into the
        benchmark until somebody fills them in. The gap is the point. Letting a
        model label its own boundary cases measures whether two models agree,
        which is a different question and a much easier one - and it would raise
        every metric in this project while adding no information at all.
        """
        criterion = self.ctx.domain.criterion(criterion_id)
        settings = dict(self.ctx.config.benchmark.get("synthesis") or {})
        count = per_criterion or int(settings.get("per_criterion", 12))

        records = list(self.ctx.domain.records.values())
        rows: list[dict[str, Any]] = []
        for index in range(count):
            record = records[index % len(records)]
            target = "PASS" if index % 2 == 0 else "FAIL"
            completion = self.ctx.runtime.call(
                "generator",
                synthesis_prompt(self.ctx.domain, criterion, record, target),
                system=SYNTHESIS_SYSTEM,
            )
            rows.append(
                {
                    "id": f"syn-{criterion_id}-{index:03d}",
                    "record_id": record.id,
                    "artefact": completion.text.strip(),
                    "intended_label": target,
                    "labels": {},
                    "rationales": {},
                    "failure_modes": {},
                    "source": "synthesised",
                    "needs_human_label": True,
                }
            )

        path = self.ctx.artifacts / f"synthesised_{criterion_id}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "criterion": criterion_id,
            "written": len(rows),
            "path": str(path),
            "next_step": (
                "A human labels each row, then the labelled rows are appended to "
                "domains/<domain>/labels.jsonl. `intended_label` is what the "
                "synthesiser was ASKED for, not a label - if you copy it into "
                "`labels` you have built a benchmark that measures model-model "
                "agreement."
            ),
        }
