"""Runs the pipeline for one or many logical dates against a simulated source.

A `Scenario` owns one source, one raw store and one warehouse, so each naive
variant and each proof runs in its own sandbox. `run_day` is the only place that
moves the simulated source forward. The pipeline sees the result as a SQLite
connection, the way it would see a real database.
"""
import shutil
from dataclasses import dataclass
from datetime import date, timedelta

from warehouse.clock import SystemClock
from warehouse.dag import RunContext, TransientError
from warehouse.land import MemoryStore, ParquetStore
from warehouse.pipeline import Env, build_dag
from warehouse.source.shop import Shop
from warehouse.warehouse import Warehouse


@dataclass
class DayRun:
    run_id: str
    run_date: date
    results: list
    xcom: dict

    @property
    def ok(self):
        return all(r.status == "success" for r in self.results)

    def status(self, task):
        return next(r.status for r in self.results if r.task == task)

    def failed_task(self):
        return next((r for r in self.results if r.status == "failed"), None)

    def skipped(self):
        return [r.task for r in self.results if r.status == "skipped"]


def fail_first(task_fn, attempts):
    """Wrap a task so its first `attempts` tries raise a transient error."""
    def wrapped(ctx):
        if ctx.attempt <= attempts:
            raise TransientError("connection reset by peer (simulated)")
        return task_fn(ctx)
    return wrapped


class Scenario:
    def __init__(self, cfg, days, opts, root=None, clock=None, sleep=None):
        """`root` is a folder for a disk-backed run. None keeps everything in memory."""
        self.cfg, self.days, self.opts = cfg, days, opts
        self.start = date.fromisoformat(cfg["source"]["start_date"])
        self.sleep = sleep or (lambda seconds: None)
        self.published_by_day = {}   # day -> revenue rows right after that day's run
        break_opts = {"volume_keep_every": cfg["demo"]["break_volume_keep_every"],
                      "null_modulus": cfg["demo"]["break_null_modulus"]}
        if root is None:
            self.root = None
            self.store, wh_path, src_path = MemoryStore(), ":memory:", ":memory:"
        else:
            self.root = root
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            self.store = ParquetStore(root / "raw")
            wh_path, src_path = root / "warehouse.duckdb", root / "source.db"
        self.shop = Shop(src_path, days, break_opts)
        self.wh = Warehouse(wh_path)
        self.env = Env(cfg, opts, self.shop, self.store, self.wh, clock or SystemClock())

    def set_options(self, opts):
        """Change the pipeline options between days, as a config change would."""
        self.opts = self.env.opts = opts

    def date_of(self, day):
        return self.start + timedelta(days=day - 1)

    def day_of(self, run_date):
        return (run_date - self.start).days + 1

    def run_day(self, day, break_kind=None, faults=None):
        """Advance the source to `day`, then run the DAG once for that logical date."""
        run_date = self.date_of(day)
        self.shop.advance_to(day, break_kind)
        dag = build_dag(self.opts)
        for task in dag.tasks:
            if faults and task.name in faults:
                task.fn = fail_first(task.fn, faults[task.name])
        run_id = self.wh.next_run_id(run_date)
        ctx = RunContext(run_id, run_date, self.env)
        seq = [0]

        def record(result):
            seq[0] += 1
            self.wh.record_task(run_id, run_date, seq[0], result)

        results = dag.run(ctx, record, self.sleep)
        self.published_by_day[day] = self.wh.revenue_by_date_region()
        return DayRun(run_id, run_date, results, ctx.xcom)

    def close(self):
        self.wh.conn.close()
        self.shop.conn.close()
