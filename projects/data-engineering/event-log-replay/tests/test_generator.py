"""The generator plants faults at exact counts, and the answer key stays out of the consume path."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from log_replay.data.generate import generate
from log_replay.faults import build_schedule
from log_replay.log.base import partition_for

SRC = Path(__file__).resolve().parents[1] / "src" / "log_replay"
# Modules allowed to name the answer key: the generator writes it, the scorer reads
# it, config holds the path. Everything else is the build path and must not see it.
KEY_ALLOWED = {"data/generate.py", "evaluation/score.py", "config.py"}


def _fingerprint(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def test_same_seed_gives_identical_stream(small_env):
    again = generate(small_env.cfg)
    assert _fingerprint(again.records) == _fingerprint(small_env.stream.records)
    assert again.answer_key == small_env.key


def test_different_seed_changes_the_stream(small_env):
    cfg = json.loads(json.dumps(small_env.cfg))
    cfg["seed"] += 1
    assert _fingerprint(generate(cfg).records) != _fingerprint(small_env.stream.records)


def test_planted_counts_are_exact(small_env):
    s, p = small_env.cfg["stream"], small_env.key["planted"]
    assert p["distinct_events"] == s["n_wallet_events"] + s["n_profile_events"]
    assert p["retried_events"] == round(s["retry_rate"] * p["distinct_events"])
    assert p["log_records"] == p["distinct_events"] + p["retried_events"]
    assert p["late_profile_updates"] == round(s["late_profile_rate"] * s["n_profile_events"])
    assert p["out_of_order_arrivals"] == p["late_profile_updates"]
    assert p["tie_accounts"] == s["tie_accounts"]


def test_every_retry_is_the_same_event_id_written_twice(small_env):
    counts = Counter(r["event_id"] for r in small_env.stream.records)
    twice = {e for e, n in counts.items() if n == 2}
    assert twice == set(small_env.key["planted"]["retried_event_ids"])
    assert max(counts.values()) == 2
    attempts = Counter((r["event_id"], r["producer_attempt"]) for r in small_env.stream.records)
    assert all(attempts[(e, 0)] == 1 and attempts[(e, 1)] == 1 for e in twice)


def test_the_retry_lands_later_in_the_same_partition(small_env):
    log, n = small_env.log, small_env.cfg["stream"]["n_partitions"]
    where = {}
    for p in range(n):
        for rec in log.read(p, 0, log.end_offset(p)):
            where.setdefault(rec.value["event_id"], []).append((p, rec.offset, rec.value["producer_attempt"]))
    for event_id in small_env.key["planted"]["retried_event_ids"]:
        (p0, o0, a0), (p1, o1, a1) = where[event_id]
        assert p0 == p1 and o1 > o0 and (a0, a1) == (0, 1)


def test_partitioning_is_by_account_and_stable(small_env):
    log, n = small_env.log, small_env.cfg["stream"]["n_partitions"]
    for p in range(n):
        for rec in log.read(p, 0, log.end_offset(p)):
            assert partition_for(rec.key, n) == p
    assert partition_for("1042", 3) == partition_for("1042", 3)


def test_answer_key_balance_applies_each_event_id_once(small_env):
    seen, balance = set(), Counter()
    for r in small_env.stream.records:
        if r["type"] != "wallet_txn" or r["event_id"] in seen:
            continue
        seen.add(r["event_id"])
        balance[r["account_id"]] += r["amount_cents"] if r["kind"] == "deposit" else -r["amount_cents"]
    for account, cents in small_env.key["balances"].items():
        assert balance[int(account)] == cents


def test_answer_key_profile_is_newest_by_time_then_version(small_env):
    best = {}
    for r in small_env.stream.records:
        if r["type"] == "profile_updated":
            k = (r["event_time"], r["version"])
            if r["account_id"] not in best or k > best[r["account_id"]][0]:
                best[r["account_id"]] = (k, r)
    for account, (_, r) in best.items():
        assert small_env.key["profiles"][str(account)] == {
            "tier": r["tier"], "daily_limit_cents": r["daily_limit_cents"]}


def test_ties_exist_and_version_breaks_them(small_env):
    by_account = {}
    for r in small_env.stream.records:
        if r["type"] == "profile_updated" and r["producer_attempt"] == 0:
            by_account.setdefault(r["account_id"], []).append(r)
    tied = [a for a, rs in by_account.items()
            if len(rs) >= 2 and len({x["event_time"] for x in rs}) < len(rs)]
    assert len(tied) >= small_env.cfg["stream"]["tie_accounts"]


def test_the_featured_account_shows_all_three_faults(small_env):
    f = small_env.manifest["featured"]
    mine = [r for r in small_env.stream.records if r["account_id"] == f["account_id"]]
    retried = set(small_env.key["planted"]["retried_event_ids"])
    assert any(r["event_id"] in retried and r["type"] == "wallet_txn" for r in mine)
    newest = 0
    out_of_order = False
    for r in mine:
        if r["type"] == "profile_updated":
            out_of_order |= r["version"] < newest
            newest = max(newest, r["version"])
    assert out_of_order
    assert (f["partition"], f["crash_offset"]) in small_env.crashes
    rec = small_env.log.read(f["partition"], f["crash_offset"], 1)[0]
    assert rec.value["event_id"] == f["crash_event_id"] and rec.key == str(f["account_id"])


def test_crash_schedule_is_seeded_and_pinned():
    a = build_schedule([1000, 1000], 100, 0.1, seed=7, pin=(0, 555))
    assert a == build_schedule([1000, 1000], 100, 0.1, seed=7, pin=(0, 555))
    assert a != build_schedule([1000, 1000], 100, 0.1, seed=8, pin=(0, 555))
    assert (0, 555) in a and build_schedule([1000], 0, 0.1, seed=7) == []


# ------------------------------------------------- the answer key stays out of the build path

def test_only_the_generator_scorer_and_config_name_the_answer_key():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text(encoding="utf-8").lower()
        if rel not in KEY_ALLOWED and "answer_key" in text:
            offenders.append(rel)
    assert not offenders, f"answer key named outside the scorer: {offenders}"


def test_consume_path_never_imports_the_scorer_or_generator():
    banned = {"evaluation", "data"}
    for rel in ["runner.py", "faults.py", "log/base.py", "log/filelog.py",
                "sinks/sql.py", "sinks/strategies.py"]:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]] + [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            assert not banned & set(names), f"{rel} imports {banned & set(names)}"


def test_consume_runs_with_the_scorer_disabled(small_env, monkeypatch):
    from log_replay import experiments

    def boom(*a, **k):
        raise AssertionError("the consume path read the answer key")

    monkeypatch.setattr(experiments, "load_key", boom)
    monkeypatch.setattr(experiments, "score_sink", boom)
    sink, stats = experiments.consume(small_env.cfg, small_env.log, "atomic", small_env.crashes)
    assert stats.crashes == len(small_env.crashes)
