"""A small DAG runner. About what Airflow gives you for this project, no more.

* Tasks declare dependencies with `>>`, like `a >> b >> [c, d]`.
* Every task receives the same context, which carries the logical `run_date`.
  Tasks derive every date from it, never from the wall clock.
* A task that raises `TransientError` is retried with exponential backoff.
  Any other exception fails the task at once. Retrying a failed data check
  would only fail again.
* A task whose upstream did not succeed is SKIPPED, not run. This is the
  Airflow `all_success` trigger rule, the default.
* Each attempt outcome goes to a `record` callback, which the pipeline points
  at the `ops.runs` table.

Airflow mapping. `DAG.run(run_date, ...)` is one DAG run with a `logical_date`.
`retries`, `retry_delay` and `retry_exponential_backoff` are the same ideas as
the arguments here. Status "skipped" here is Airflow's `upstream_failed`.
"""
import time
from dataclasses import dataclass, field


class TransientError(Exception):
    """A failure worth retrying, such as a dropped connection."""


class Task:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn
        self.upstream = []

    def __rshift__(self, other):
        for t in (other if isinstance(other, list) else [other]):
            t.upstream.append(self)
        return other

    def __rrshift__(self, others):
        # [a, b] >> c
        for t in others:
            t >> self
        return self


@dataclass
class TaskResult:
    task: str
    status: str            # success, failed, skipped
    attempts: int = 0
    error: str = ""


@dataclass
class RunContext:
    run_id: str
    run_date: object
    env: object
    attempt: int = 0
    xcom: dict = field(default_factory=dict)


class DAG:
    def __init__(self, name, retries=2, retry_delay=0.05, backoff=2.0):
        self.name, self.retries = name, retries
        self.retry_delay, self.backoff = retry_delay, backoff
        self.tasks = []

    def add(self, name, fn):
        task = Task(name, fn)
        self.tasks.append(task)
        return task

    def order(self):
        """Dependency order, ties broken by declaration order, so runs are repeatable."""
        done, ordered, pending = set(), [], list(self.tasks)
        while pending:
            ready = [t for t in pending if all(u.name in done for u in t.upstream)]
            if not ready:
                raise ValueError("cycle in DAG: " + ", ".join(t.name for t in pending))
            ordered.append(ready[0])
            done.add(ready[0].name)
            pending.remove(ready[0])
        return ordered

    def run(self, ctx, record, sleep=time.sleep):
        """Run every task once in order. `record(result)` is called per task."""
        status, results = {}, []
        for task in self.order():
            if any(status[u.name] != "success" for u in task.upstream):
                res = TaskResult(task.name, "skipped")
            else:
                res = self._run_task(task, ctx, sleep)
            status[task.name] = res.status
            results.append(res)
            record(res)
        return results

    def _run_task(self, task, ctx, sleep):
        delay = self.retry_delay
        for attempt in range(1, self.retries + 2):
            ctx.attempt = attempt
            try:
                ctx.xcom[task.name] = task.fn(ctx)
                return TaskResult(task.name, "success", attempt)
            except TransientError as exc:
                if attempt > self.retries:
                    return TaskResult(task.name, "failed", attempt, f"TransientError: {exc}")
                sleep(delay)
                delay *= self.backoff
            except Exception as exc:  # noqa: BLE001 - a task may fail for any reason
                return TaskResult(task.name, "failed", attempt, f"{type(exc).__name__}: {exc}")
