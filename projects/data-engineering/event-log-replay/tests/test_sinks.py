"""Sink behaviour on the small stream, including a mutation test for each protection."""
import pytest

from log_replay.evaluation.score import score_sink
from log_replay.experiments import consume, run_and_score
from log_replay.log.base import Consumer
from log_replay.runner import run_consumer
from log_replay.sinks import sql
from log_replay.sinks.strategies import (AtomicOffsetOnlySink, AtomicSink, Crash,
                                         DedupSeparateTxnSink, NaiveSink)


_MEMO = {}


def run(env, strategy, crashes=None, batch=None):
    """Runs are deterministic, so one run per (strategy, crashes, batch) serves every test."""
    k = (id(env), strategy, None if crashes is None else tuple(crashes), batch)
    if k not in _MEMO:
        crashes = env.crashes if crashes is None else crashes
        _MEMO[k] = run_and_score(env.cfg, env.log, strategy, crashes, batch, env.key)
    return _MEMO[k]


def test_without_crashes_atomic_is_exact(small_env):
    r = run(small_env, "atomic", crashes=[])
    assert r["score"]["accounts_wrong"] == 0 and r["score"]["profiles_wrong"] == 0
    assert r["score"]["applied_twice"] == 0 and r["score"]["events_lost"] == 0


def test_naive_without_crashes_is_wrong_only_by_the_producer_retries(small_env):
    r = run(small_env, "naive", crashes=[])
    assert r["score"]["applied_twice"] == small_env.key["planted"]["retried_wallet_events"]
    assert r["score"]["events_lost"] == 0 and r["score"]["net_drift_cents"] > 0


def test_naive_redelivery_applies_the_handled_prefix_twice(small_env):
    clean = run(small_env, "naive", crashes=[])["score"]["applied_twice"]
    crashed = run(small_env, "naive")
    assert crashed["score"]["applied_twice"] > clean
    assert crashed["stats"]["crashes"] == len(small_env.crashes)
    assert crashed["stats"]["redelivered"] > 0


def test_commit_first_loses_events_and_never_redelivers(small_env):
    r = run(small_env, "commit_first")
    assert r["score"]["events_lost"] > 0 and r["stats"]["redelivered"] == 0


def test_dedup_in_a_separate_txn_fixes_retries_but_not_crashes(small_env):
    naive = run(small_env, "naive")["score"]
    dedup = run(small_env, "dedup_separate_txn")["score"]
    assert 0 < dedup["applied_twice"] < naive["applied_twice"]
    assert dedup["events_lost"] == 0


def test_atomic_is_exact_with_crashes_at_every_batch_size(small_env):
    for batch in (5, 20, 500):
        s = run(small_env, "atomic", batch=batch)["score"]
        assert (s["accounts_wrong"], s["profiles_wrong"], s["events_lost"], s["applied_twice"]) == (0, 0, 0, 0), batch


def test_atomic_result_does_not_depend_on_batch_size(small_env):
    def state(batch):
        sink, _ = consume(small_env.cfg, small_env.log, "atomic", small_env.crashes, batch)
        return (sink.con.execute("SELECT * FROM balances ORDER BY 1").fetchall(),
                sink.con.execute("SELECT * FROM profiles ORDER BY 1").fetchall())
    assert state(10) == state(250)


def test_redelivery_grows_with_batch_size(small_env):
    redelivered = [run(small_env, "atomic", batch=b)["stats"]["redelivered"] for b in (5, 20, 200)]
    assert redelivered == sorted(redelivered) and redelivered[0] < redelivered[-1]


def test_profile_policies_rank_as_taught(small_env):
    lww = run(small_env, "atomic_last_arrived_wins", crashes=[])["score"]["profiles_wrong"]
    time_only = run(small_env, "atomic_time_only_guard", crashes=[])["score"]["profiles_wrong"]
    guarded = run(small_env, "atomic", crashes=[])["score"]["profiles_wrong"]
    assert lww > time_only > guarded == 0


# ------------------------------------------ what a crash leaves behind, record by record

def _first_batch(env, n=10):
    return Consumer(env.log, "unit").poll(n)


def test_atomic_crash_leaves_no_trace_in_the_sink(small_env):
    sink = AtomicSink(sql.connect(), "guarded_upsert")
    batch = _first_batch(small_env)
    with pytest.raises(Crash):
        sink.handle(Consumer(small_env.log, "unit"), batch, crash_at=6)
    for table in ("balances", "profiles", "processed_events", "consumer_offsets", "effect_audit"):
        assert sink.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_atomic_commit_stores_effects_ids_and_offsets_together(small_env):
    sink = AtomicSink(sql.connect(), "guarded_upsert")
    batch = _first_batch(small_env)
    sink.handle(Consumer(small_env.log, "unit"), batch, crash_at=None)
    assert sink.con.execute("SELECT COUNT(*) FROM processed_events").fetchone()[0] == len(batch)
    offsets = dict(sink.con.execute("SELECT * FROM consumer_offsets").fetchall())
    assert sum(offsets.values()) == len(batch)


def test_dedup_separate_crash_leaves_effects_without_ids(small_env):
    sink = DedupSeparateTxnSink(sql.connect(), "guarded_upsert")
    batch = _first_batch(small_env)
    with pytest.raises(Crash):
        sink.handle(Consumer(small_env.log, "unit"), batch, crash_at=6)
    assert sink.con.execute("SELECT COUNT(*) FROM processed_events").fetchone()[0] == 0
    assert sink.con.execute("SELECT COUNT(*) FROM effect_audit").fetchone()[0] > 0


def test_naive_crash_leaves_effects_and_no_commit(small_env):
    log, sink = small_env.log, NaiveSink(sql.connect(), "last_arrived_wins")
    log.reset_group("unit-naive")
    consumer = Consumer(log, "unit-naive")
    batch = consumer.poll(10)
    with pytest.raises(Crash):
        sink.handle(consumer, batch, crash_at=6)
    assert sink.con.execute("SELECT COUNT(*) FROM effect_audit").fetchone()[0] > 0
    assert Consumer(log, "unit-naive").position == {0: 0, 1: 0, 2: 0}


# --------------------------------------------------------------- mutation tests
# Each protection is real only if removing it turns the scoreboard red.

class OffsetOutsideTxnAtomic(AtomicSink):
    """Atomic with the offset stored in its own transaction after the effects."""
    def handle(self, consumer, batch, crash_at):
        con = self.con
        done = batch if crash_at is None else batch[: crash_at + 1]
        sql.load_batch(con, done)
        con.execute("BEGIN")
        sql.select_fresh(con)
        sql.apply_deltas(con, sql.FRESH)
        sql.apply_profiles(con, sql.PROFILE, self.policy)
        sql.mark_processed(con)
        con.execute("COMMIT")
        if crash_at is not None:
            raise Crash(crash_at)  # effects and ids durable, offset not stored
        con.execute("BEGIN")
        sql.store_offsets(con)
        con.execute("COMMIT")


def _run_sink(env, sink_cls, policy="guarded_upsert"):
    sink = sink_cls(sql.connect(), policy)
    run_consumer(env.log, sink, "mutant", env.cfg["consumer"]["batch_size"], env.crashes)
    return score_sink(sink.con, env.key)


def test_mutation_removing_the_id_table_leaves_the_producer_retries(small_env):
    """Offsets in the transaction stop crash duplicates. They cannot see a producer retry."""
    s = _run_sink(small_env, AtomicOffsetOnlySink)
    assert s["applied_twice"] == small_env.key["planted"]["retried_wallet_events"]
    assert s["events_lost"] == 0 and s["accounts_wrong"] > 0


def test_mutation_stale_offset_is_harmless_when_the_ids_commit_with_the_effects(small_env):
    """The id table carries correctness. The offset only saves re-reading."""
    assert _run_sink(small_env, OffsetOutsideTxnAtomic)["accounts_wrong"] == 0


def test_mutation_swapping_the_profile_guard_breaks_profiles(small_env):
    s = _run_sink(small_env, AtomicSink, policy="last_arrived_wins")
    assert s["profiles_wrong"] > 0 and s["accounts_wrong"] == 0
