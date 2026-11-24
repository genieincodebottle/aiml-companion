"""Saga / compensating-transaction support for side-effecting commits.

A disposition commit is not one atomic write - it may record a case, notify a
downstream system, and update an index. If a later sub-action fails, the earlier ones
must be undone or the world is left inconsistent. The saga records a compensation for
each completed sub-action and runs them in reverse on failure.

This is the distributed-systems answer to "there is no distributed transaction across
an LLM, a database, and a message bus": you compensate, you don't roll back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class _Step:
    name: str
    compensate: Callable[[], None]


class SagaError(RuntimeError):
    def __init__(self, failed_step: str, cause: Exception) -> None:
        super().__init__(f"saga failed at '{failed_step}': {cause}")
        self.failed_step = failed_step
        self.cause = cause


@dataclass
class Saga:
    """Accumulates forward actions + their compensations; unwinds on failure."""

    _done: list[_Step] = field(default_factory=list)
    compensated: list[str] = field(default_factory=list)

    def run(self, name: str, forward: Callable[[], None], compensate: Callable[[], None]) -> None:
        """Run a forward sub-action; on failure, compensate everything so far."""
        try:
            forward()
        except Exception as exc:  # noqa: BLE001 - trigger compensation then surface
            self.unwind()
            raise SagaError(name, exc) from exc
        self._done.append(_Step(name=name, compensate=compensate))

    def unwind(self) -> None:
        """Run compensations in reverse order (best-effort; never raises)."""
        for step in reversed(self._done):
            try:
                step.compensate()
                self.compensated.append(step.name)
            except Exception:  # noqa: BLE001 - compensation is best-effort
                pass
        self._done.clear()
