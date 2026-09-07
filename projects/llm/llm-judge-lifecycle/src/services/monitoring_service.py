"""Phase IV service: read a week's ratings, check the band, decide what happens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..monitoring import augment_examples, detect_drift, load_rated_items
from ._context import Context, build_context


class MonitoringService:
    def __init__(self, context: Context | None = None) -> None:
        self.ctx = context or build_context()

    def week_path(self, week: int) -> Path:
        return self.ctx.domain.path / "hitl" / f"week_{week:02d}.jsonl"

    def available_weeks(self) -> list[int]:
        directory = self.ctx.domain.path / "hitl"
        if not directory.exists():
            return []
        weeks: list[int] = []
        for path in sorted(directory.glob("week_*.jsonl")):
            try:
                weeks.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return weeks

    def check(self, week: int) -> dict[str, Any]:
        rows = _read_jsonl(self.week_path(week))
        if not rows:
            raise FileNotFoundError(
                f"no rated sample for week {week} at {self.week_path(week)}. "
                f"Available weeks: {self.available_weeks()}"
            )

        config = dict(self.ctx.config.monitoring)
        items = load_rated_items(
            rows,
            self.ctx.domain,
            week=week,
            lookback_weeks=int(config.get("new_item_lookback_weeks", 2)),
        )

        by_criterion: dict[str, list] = {}
        for item in items:
            by_criterion.setdefault(item.criterion, []).append(item)

        reports = [
            detect_drift(rows_, config, week=week, criterion=criterion).as_dict()
            for criterion, rows_ in sorted(by_criterion.items())
        ]
        alerting = [r for r in reports if r["alert"]]

        return {
            "week": week,
            "n_rated": len(items),
            "n_new_items": sum(1 for i in items if i.is_new_item),
            "reports": reports,
            "alert": bool(alerting),
            "action": self._action(alerting, config),
        }

    def _action(self, alerting: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        if not alerting:
            return {
                "retune": False,
                "note": "Every metric sits inside the band. No action. Note that "
                "the loop still matters: today's agreement says little about "
                "tomorrow's, because the catalogue and the generator keep moving.",
            }

        auto = bool(config.get("auto_retune_on_drift", True))
        deploy = bool(config.get("auto_deploy_retuned_rubric", False))
        criteria = sorted({r["criterion"] for r in alerting})
        return {
            "retune": auto,
            "criteria": criteria,
            "commands": [f"python run.py tune --criterion {c}" for c in criteria],
            "deploy": deploy,
            "note": (
                "A drift alert triggers re-tuning on the augmented benchmark. "
                "The new rubric is STAGED and a human reads the diff before it "
                "ships. auto_deploy_retuned_rubric is off, and should stay off: "
                "an unsupervised system that re-tunes and self-deploys is "
                "editing its own success criteria, and it will eventually decide "
                "it is doing well."
            )
            if not deploy
            else (
                "auto_deploy_retuned_rubric is ON. The judge will change what it "
                "rejects with no human in the path. This is not recommended."
            ),
        }

    def augment(self, week: int) -> dict[str, Any]:
        """Append the week's rated sample to the benchmark, preserving balance.

        Written to artifacts/ for review rather than straight into
        domains/<domain>/labels.jsonl. The benchmark is the ground truth every
        other number in the project is measured against, and a pipeline that can
        rewrite it without a human reading the diff is a pipeline that can move
        the goalposts and pass.
        """
        rows = _read_jsonl(self.week_path(week))
        items = load_rated_items(rows, self.ctx.domain, week=week)
        target = float(self.ctx.config.benchmark.get("target_fail_fraction", 0.5))
        new_rows = augment_examples(items, target)

        path = self.ctx.artifacts / f"benchmark_append_week_{week:02d}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return {
            "week": week,
            "rated": len(items),
            "appendable": len(new_rows),
            "dropped_for_balance": len(items) - len(new_rows),
            "path": str(path),
            "next_step": (
                "Review, then append to domains/"
                f"{self.ctx.domain.name}/labels.jsonl. Content-addressed splits "
                "mean the existing examples do not move when you do - so last "
                "week's test set stays the test set, and week-over-week "
                "comparisons remain comparable."
            ),
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows
