"""Fixtures. Two sizes of the same project.

`small_env` is a few thousand events in a temp folder. It exercises the logic in
well under a second per run and touches nothing on disk that matters.

`full_env` is the real config (50,000 events). It is regenerated into data/ at
the start of the session, which is safe because the same seed gives
byte-identical files, and the headline numbers in test_lessons.py come from it.
"""
import json
from types import SimpleNamespace

import pytest

from log_replay.config import LOG_DIR, MANIFEST_PATH, load_config
from log_replay.data.generate import generate, write_stream
from log_replay.experiments import crash_schedule, make_job, run_jobs
from log_replay.log.filelog import FileLog


def small_config() -> dict:
    cfg = load_config()
    cfg["stream"].update(n_accounts=80, n_wallet_events=1200, n_profile_events=300,
                         retry_rate=0.03, late_profile_rate=0.10, tie_accounts=5,
                         retry_max_gap=15, late_max_gap=20)
    cfg["crashes"].update(every=40)
    cfg["consumer"]["batch_size"] = 20
    return cfg


def _build(cfg, root):
    s = cfg["stream"]
    stream = generate(cfg)
    log = FileLog(root, s["topic"], s["n_partitions"], create=True)
    write_stream(stream, cfg, log, root / "key.json", root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return SimpleNamespace(cfg=cfg, stream=stream, key=stream.answer_key, manifest=manifest,
                           log=FileLog(root, s["topic"], s["n_partitions"]),
                           crashes=crash_schedule(cfg, manifest))


@pytest.fixture(scope="session")
def small_env(tmp_path_factory):
    return _build(small_config(), tmp_path_factory.mktemp("small"))


@pytest.fixture(scope="session")
def full_env():
    cfg = load_config()
    stream = generate(cfg)
    s = cfg["stream"]
    log = FileLog(LOG_DIR, s["topic"], s["n_partitions"], create=True)
    write_stream(stream, cfg, log)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return SimpleNamespace(cfg=cfg, stream=stream, key=stream.answer_key, manifest=manifest,
                           log=FileLog(LOG_DIR, s["topic"], s["n_partitions"]),
                           crashes=crash_schedule(cfg, manifest))


@pytest.fixture(scope="session")
def scoreboard(full_env):
    """Every compare run, in parallel, keyed by label."""
    e = full_env
    jobs = [make_job(e.cfg, "naive", [], label="naive (no crashes)")]
    jobs += [make_job(e.cfg, n, e.crashes, export=True) for n in
             ["naive", "commit_first", "dedup_separate_txn", "atomic_offset_only", "atomic",
              "atomic_last_arrived_wins", "atomic_time_only_guard"]]
    return {r["label"]: r for r in run_jobs(jobs, workers=6)}


@pytest.fixture(scope="session")
def replays(full_env):
    e = full_env
    names = ["naive", "commit_first", "dedup_separate_txn", "atomic"]
    res = run_jobs([make_job(e.cfg, n, e.crashes, kind="replay") for n in names], workers=4)
    return {r["strategy"]: r for r in res}


@pytest.fixture(scope="session")
def batch_runs(full_env):
    e = full_env
    sizes = e.cfg["experiments"]["batch_sizes"]
    res = run_jobs([make_job(e.cfg, "atomic", e.crashes, label=f"b{b}", batch=b) for b in sizes],
                   workers=4)
    return dict(zip(sizes, res))
