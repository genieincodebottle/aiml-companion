"""The same sinks against a real broker. Skipped unless KAFKA_BOOTSTRAP is set.

    docker compose up -d
    KAFKA_BOOTSTRAP=localhost:19092 pytest -m kafka
"""
import os

import pytest

pytestmark = [pytest.mark.kafka,
              pytest.mark.skipif(not os.environ.get("KAFKA_BOOTSTRAP"),
                                 reason="set KAFKA_BOOTSTRAP to run against a broker")]


@pytest.fixture(scope="module")
def kafka_log(full_env):
    from log_replay.log.kafkalog import KafkaLog
    s = full_env.cfg["stream"]
    log = KafkaLog(os.environ["KAFKA_BOOTSTRAP"], s["topic"], s["n_partitions"])
    log.recreate_topic()
    for p in range(full_env.log.n_partitions):
        for rec in full_env.log.read(p, 0, full_env.log.end_offset(p)):
            log.append(p, rec.key, rec.value)
    log.flush()
    return log


def test_the_broker_holds_the_same_partitions(full_env, kafka_log):
    for p in range(full_env.log.n_partitions):
        assert kafka_log.end_offset(p) == full_env.log.end_offset(p)
        ours = full_env.log.read(p, 0, 25)
        theirs = kafka_log.read(p, 0, 25)
        assert [(r.offset, r.key, r.value) for r in ours] == [(r.offset, r.key, r.value) for r in theirs]


def test_kafka_numbers_equal_the_file_backend_numbers(full_env, kafka_log, scoreboard):
    from log_replay.experiments import make_job, run_jobs
    boot = os.environ["KAFKA_BOOTSTRAP"]
    jobs = [make_job(full_env.cfg, n, full_env.crashes, backend="kafka", bootstrap=boot)
            for n in ("naive", "commit_first", "atomic")]
    for r in run_jobs(jobs, workers=3):
        mine = scoreboard[r["strategy"]]
        assert r["score"] == mine["score"], r["strategy"]
        assert r["stats"] == mine["stats"], r["strategy"]
