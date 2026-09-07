"""The thing under test. Writes the artefact; never judges it.

Kept in its own module with its own role config so it can be pointed at a
different provider from the judge. That separation is the only way to answer
"is the judge measuring quality, or measuring how much this output looks like
its own writing?" - and a project where the two share one client cannot ask the
question at all.
"""

from __future__ import annotations

from .domain import Domain, Record
from .prompts import generator_prompt
from .runtime import Runtime


class Generator:
    def __init__(self, runtime: Runtime, domain: Domain) -> None:
        self.runtime = runtime
        self.domain = domain
        self.system = domain.generation.get("system", "").strip() or None

    def write(
        self, record: Record, *, critique: str | None = None, attempt: int = 0
    ) -> str:
        completion = self.runtime.call(
            "generator",
            generator_prompt(self.domain, record, critique=critique, attempt=attempt),
            system=self.system,
        )
        return completion.text.strip()
