"""Background jobs: ingestion and evaluation.

Both take minutes, so both run in a worker thread behind a status object rather
than blocking a request. A two-minute HTTP request is not merely slow - it gets
killed by a proxy, a load balancer or an impatient browser, and the work carries
on with nobody left to receive the result.

One job at a time per kind, deliberately. Two concurrent ingestions against one
graph interleave their MERGEs and produce a graph neither intended, and the
result is not an error - it is a plausible-looking graph that is quietly wrong.
The lock is the fix here; a real deployment uses a queue with a single consumer,
which is the same idea with durability added.

See `src/services/qa.py` for the layering rationale. Nothing here imports
fastapi: `start()` raises RuntimeError when a job is already running, and the
routing layer decides that means 409.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from ..config import Config
from ..guardrails import GuardrailEngine


class JobBusy(RuntimeError):
    """A job of this kind is already running."""


class Job:
    """State for one background job, guarded by a lock because the worker
    thread writes it while request threads read it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self.running = False
        self.message = ""
        self.progress = 0.0
        self.report: dict[str, Any] | None = None
        self.error: str | None = None

    def _begin(self) -> None:
        with self._lock:
            if self.running:
                raise JobBusy(
                    f"A {self.name} job is already running. Poll its status "
                    "endpoint until it finishes."
                )
            self.running = True
            self.message = "starting"
            self.progress = 0.0
            self.report = None
            self.error = None

    def update(self, message: str, progress: float) -> None:
        with self._lock:
            self.message = message
            self.progress = min(max(progress, 0.0), 1.0)

    def _finish(self, report: dict[str, Any] | None = None,
                error: str | None = None) -> None:
        with self._lock:
            self.running = False
            self.report = report
            self.error = error
            self.message = "failed" if error else "complete"
            self.progress = 1.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"running": self.running, "message": self.message,
                    "progress": self.progress, "report": self.report,
                    "error": self.error}

    def run_in_background(self, work: Callable[[], dict[str, Any]],
                          *, on_success: Callable[[], None] | None = None,
                          on_error: Callable[[Exception], None] | None = None,
                          ) -> dict[str, Any]:
        self._begin()

        def target() -> None:
            try:
                report = work()
                if on_success:
                    on_success()
                self._finish(report=report)
            except Exception as exc:  # noqa: BLE001 - a worker must never die silently
                if on_error:
                    on_error(exc)
                self._finish(error=f"{type(exc).__name__}: {exc}\n\n"
                                   f"{traceback.format_exc(limit=3)}")

        threading.Thread(target=target, daemon=True, name=self.name).start()
        return self.snapshot()


class JobService:
    def __init__(self, *, config: Config, guard: GuardrailEngine,
                 on_graph_changed: Callable[[], None] | None = None) -> None:
        self.config = config
        self.guard = guard
        self.on_graph_changed = on_graph_changed
        self.ingest_job = Job("ingest")
        self.eval_job = Job("eval")

    # --------------------------------------------------------------- ingest
    def start_ingest(self, *, reset: bool, caller: str = "local") -> dict[str, Any]:
        from ..ingest.pipeline import ingest

        self.guard.audit.write("ingest_started", caller=caller, reset=reset)

        def work() -> dict[str, Any]:
            report = ingest(reset=reset, config=self.config,
                            progress=self.ingest_job.update)
            self.guard.audit.write("ingest_complete", **report.as_dict())
            return report.as_dict()

        def success() -> None:
            # The entity-name cache backs the output validator. Leaving it stale
            # after ingestion means the validator flags newly-added, entirely
            # real suppliers as hallucinations.
            if self.on_graph_changed:
                self.on_graph_changed()

        return self.ingest_job.run_in_background(
            work, on_success=success,
            on_error=lambda exc: self.guard.audit.write("ingest_failed",
                                                        error=str(exc)),
        )

    def ingest_status(self) -> dict[str, Any]:
        return self.ingest_job.snapshot()

    # ----------------------------------------------------------------- eval
    def start_eval(self, *, strategies: list[str] | None,
                   question_ids: list[str] | None, judge: bool,
                   caller: str = "local") -> dict[str, Any]:
        from ..evaluate import evaluate

        self.guard.audit.write("eval_started", caller=caller, judge=judge,
                               questions=question_ids)

        def work() -> dict[str, Any]:
            return evaluate(strategies=strategies, judge=judge,
                            question_ids=question_ids, config=self.config,
                            progress=self.eval_job.update)

        return self.eval_job.run_in_background(work)

    def eval_status(self) -> dict[str, Any]:
        return self.eval_job.snapshot()

    def golden_questions(self) -> list[dict[str, Any]]:
        from ..evaluate import load_questions
        return load_questions(self.config.golden_questions)
