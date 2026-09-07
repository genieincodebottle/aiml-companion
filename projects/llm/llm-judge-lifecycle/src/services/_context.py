"""Shared construction: config, domain, runtime, and the rubric store.

The rubric store is the interesting part. A tuned rubric is the deployable
artefact of Phase II - the thing that changes what the judge does - so it is
versioned on disk rather than held in memory, and Phase III reads whatever is
currently staged.

Loading always falls back to the seed rubric from ``domain.yaml`` when nothing
is staged. That is what makes the whole project runnable in any order: you can
serve, or monitor, before you have ever tuned, and the judge is the guideline
its raters work to. A missing artefact should degrade the system to its
defensible starting point, not to an exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config, as_offline, get_config
from ..domain import Domain, get_domain
from ..runtime import Runtime


@dataclass
class Context:
    config: Config
    domain: Domain
    runtime: Runtime

    @property
    def artifacts(self) -> Path:
        path = self.config.root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------- rubrics
    def rubric_path(self, criterion_id: str, *, staged: bool = False) -> Path:
        stage = "staged" if staged else "live"
        directory = self.artifacts / "rubrics" / self.domain.name / stage
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{criterion_id}.md"

    def load_rubrics(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for criterion in self.domain.criteria:
            path = self.rubric_path(criterion.id)
            out[criterion.id] = (
                path.read_text(encoding="utf-8")
                if path.exists()
                else self.domain.seed_rubric(criterion.id)
            )
        return out

    def save_rubric(self, criterion_id: str, rubric: str, *, staged: bool) -> Path:
        path = self.rubric_path(criterion_id, staged=staged)
        path.write_text(rubric, encoding="utf-8")
        return path

    # ------------------------------------------------------------ artefacts
    def write_artifact(self, name: str, payload: dict[str, Any]) -> Path:
        """Write a result file, stamped with what produced it.

        Provenance is merged in here rather than at each call site, because the
        one call site that forgets is the one whose numbers end up in a
        comparison they cannot support. The stamp records the model behind every
        role, whether the run was fully offline, and whether the generator and
        judge were the same model.
        """
        path = self.artifacts / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "domain": self.domain.name,
            "provenance": self.runtime.provenance(),
            **payload,
        }
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def build_context(
    config: Config | None = None,
    *,
    max_usd: float | None = None,
    offline: bool = False,
) -> Context:
    config = config or get_config()
    if offline:
        config = as_offline(config)
    domain = get_domain(config.domain_name)
    return Context(
        config=config, domain=domain, runtime=Runtime(config, max_usd=max_usd)
    )
