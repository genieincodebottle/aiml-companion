"""The experiments run.py exposes. Every one runs the same stream and the same crash schedule."""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from .config import ARTIFACTS_DIR, LOG_DIR, MANIFEST_PATH
from .evaluation.score import load_key, score_sink
from .faults import build_schedule
from .log.base import LogBackend
from .log.filelog import FileLog
from .runner import RunStats, run_consumer
from .sinks.sql import connect
from .sinks.strategies import SINKS, Sink

# name -> (sink class, profile policy). Balances and profiles are separate
# decisions, so the profile policy is a column of the scoreboard, not a new sink.
STRATEGIES = {
    "naive": ("naive", "last_arrived_wins"),
    "commit_first": ("commit_first", "last_arrived_wins"),
    "dedup_separate_txn": ("dedup_separate_txn", "guarded_upsert"),
    "atomic": ("atomic", "guarded_upsert"),
    "atomic_offset_only": ("atomic_offset_only", "guarded_upsert"),
    "atomic_last_arrived_wins": ("atomic", "last_arrived_wins"),
    "atomic_time_only_guard": ("atomic", "time_only_guard"),
}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def open_log(cfg: dict, backend: str = "file", bootstrap: str | None = None) -> LogBackend:
    s = cfg["stream"]
    if backend == "file":
        return FileLog(LOG_DIR, s["topic"], s["n_partitions"])
    from .log.kafkalog import KafkaLog  # optional dependency, imported only when asked for
    bootstrap = bootstrap or os.environ.get(cfg["kafka"]["bootstrap_env"])
    if not bootstrap:
        raise SystemExit(f"Set {cfg['kafka']['bootstrap_env']} (for example localhost:19092) "
                         "or pass --bootstrap.")
    return KafkaLog(bootstrap, s["topic"], s["n_partitions"])


def crash_schedule(cfg: dict, manifest: dict, every: int | None = None) -> list:
    """Crash positions for this log. every=0 means a clean run with no crashes."""
    every = cfg["crashes"]["every"] if every is None else every
    f = manifest["featured"]
    return build_schedule(manifest["partition_lengths"], every, cfg["crashes"]["jitter"],
                          cfg["seed"], pin=(f["partition"], f["crash_offset"]))


def make_sink(strategy: str, con=None) -> Sink:
    sink_name, policy = STRATEGIES[strategy]
    return SINKS[sink_name](con if con is not None else connect(), policy)


def group_for(cfg: dict, label: str) -> str:
    """One consumer group per run. On a shared broker two runs in one group would steal each other's offsets."""
    return f"{cfg['consumer']['group']}-{re.sub('[^A-Za-z0-9]+', '-', label).strip('-')}"


def consume(cfg: dict, log: LogBackend, strategy: str, crashes: list,
            batch_size: int | None = None, label: str | None = None) -> tuple[Sink, RunStats]:
    """Run one strategy over the whole log, from offset zero, into a fresh sink."""
    sink = make_sink(strategy)
    group = group_for(cfg, label or strategy)
    if sink.name != "atomic":
        log.reset_group(group)
    stats = run_consumer(log, sink, group, batch_size or cfg["consumer"]["batch_size"], crashes)
    return sink, stats


def run_and_score(cfg: dict, log: LogBackend, strategy: str, crashes: list,
                  batch_size: int | None = None, key: dict | None = None,
                  label: str | None = None) -> dict:
    key = key or load_key()
    sink, stats = consume(cfg, log, strategy, crashes, batch_size, label)
    return {"strategy": strategy, "sink": sink, "stats": asdict(stats),
            "score": score_sink(sink.con, key),
            "batch_size": batch_size or cfg["consumer"]["batch_size"], "crashes": len(crashes)}


# ---------------------------------------------------------------- replay proof

CHECKED_TABLES = {"balances": "account_id", "profiles": "account_id",
                  "processed_events": "event_id"}


def _rows(con, table: str) -> list:
    return con.execute(f"SELECT * FROM {table} ORDER BY {CHECKED_TABLES[table]}").fetchall()


def checksum(rows: list) -> str:
    """sha256 over every row in key order, so it does not depend on storage order."""
    h = hashlib.sha256()
    for row in rows:
        h.update(("|".join(str(c) for c in row) + "\n").encode("utf-8"))
    return h.hexdigest()[:16]


def rows_differing(before: list, after: list) -> int:
    a, b = {r[0]: r for r in before}, {r[0]: r for r in after}
    return sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))


def replay_proof(cfg: dict, log: LogBackend, strategy: str, crashes: list, key: dict,
                 sink: Sink | None = None, label: str | None = None) -> dict:
    """Consume with crashes, rewind to offset zero, consume everything again, diff the sink.

    Pass a sink that has already consumed the log to reuse it instead of consuming again."""
    if sink is None:
        sink, _ = consume(cfg, log, strategy, crashes, label=label)
    con = sink.con
    before = {t: _rows(con, t) for t in CHECKED_TABLES}
    score_before = score_sink(con, key)
    group = group_for(cfg, label or strategy)
    sink.rewind(log, group)
    run_consumer(log, sink, group, cfg["consumer"]["batch_size"], [])
    after = {t: _rows(con, t) for t in CHECKED_TABLES}
    score_after = score_sink(con, key)
    sums_before = {t: checksum(r) for t, r in before.items()}
    sums_after = {t: checksum(r) for t, r in after.items()}
    return {
        "strategy": strategy, "checksum_before": sums_before, "checksum_after": sums_after,
        "balances_differ": rows_differing(before["balances"], after["balances"]),
        "profiles_differ": rows_differing(before["profiles"], after["profiles"]),
        "processed_differ": rows_differing(before["processed_events"], after["processed_events"]),
        "identical": sums_before == sums_after,
        "wrong_before": score_before["accounts_wrong"], "wrong_after": score_after["accounts_wrong"],
        "net_drift_before": score_before["net_drift_cents"],
        "net_drift_after": score_after["net_drift_cents"],
    }


# --------------------------------------------------------------- sink exports

def export_sink(sink: Sink, strategy: str, score: dict) -> None:
    """Write the sink tables and effect anomalies so `inspect` can read them back."""
    out = ARTIFACTS_DIR / "sinks"
    out.mkdir(parents=True, exist_ok=True)
    for table in ("balances", "profiles"):
        path = (out / f"{strategy}_{table}.csv").as_posix()
        sink.con.execute(f"COPY (SELECT * FROM {table} ORDER BY account_id) TO '{path}' "
                         "(HEADER, DELIMITER ',')")
    lines = ["event_id,times_applied"] + [f"{e},{n}" for e, n in score["anomalies"]]
    (out / f"{strategy}_effects.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------- parallel jobs

def make_job(cfg: dict, strategy: str, crashes: list, kind: str = "score", label: str | None = None,
             batch: int | None = None, export: bool = False, backend: str = "file",
             bootstrap: str | None = None) -> dict:
    return {"kind": kind, "cfg": cfg, "backend": backend, "bootstrap": bootstrap,
            "strategy": strategy, "crashes": crashes, "batch": batch,
            "label": label or strategy, "export": export}


def run_job(job: dict) -> dict:
    """One independent run. Top level and plain-data in, plain-data out, so a worker process can run it.

    Runs share nothing but the read-only log, and each strategy has its own
    consumer group, so running them side by side cannot change a number."""
    cfg, key = job["cfg"], load_key()
    log = open_log(cfg, job["backend"], job["bootstrap"])
    if job["kind"] == "replay":
        return replay_proof(cfg, log, job["strategy"], job["crashes"], key, label=job["label"])
    r = run_and_score(cfg, log, job["strategy"], job["crashes"], job.get("batch"), key, job["label"])
    if job.get("export"):
        export_sink(r["sink"], job["label"], r["score"])
    del r["sink"]
    r["label"] = job["label"]
    return r


def run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    """Results in job order. workers=1 runs in this process, which is easier to debug."""
    if workers <= 1 or len(jobs) == 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(run_job, jobs))
