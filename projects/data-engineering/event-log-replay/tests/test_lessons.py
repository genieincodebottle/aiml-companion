"""Regression tests on the project's actual claims. Every number in the README is
asserted here at full scale (50,000 events). If a lesson stops being true the
suite goes red, so the README cannot drift from the code."""
import csv
import io
import json
from contextlib import redirect_stdout

from log_replay import cli
from log_replay.config import ARTIFACTS_DIR

# (accounts wrong, net drift cents, abs drift cents, lost, applied twice, profiles wrong, redelivered)
SCOREBOARD = {
    "naive (no crashes)": (725, 8093302, 16218846, 0, 891, 353, 0),
    "naive": (1580, 28156711, 47519463, 0, 3109, 353, 2470),
    "commit_first": (1638, -14525981, 48191323, 2442, 828, 441, 0),
    "dedup_separate_txn": (1355, 19542152, 36531892, 0, 2175, 0, 2470),
    "atomic_offset_only": (725, 8093302, 16218846, 0, 891, 0, 2470),
    "atomic": (0, 0, 0, 0, 0, 0, 2470),
}


def row(r):
    s = r["score"]
    return (s["accounts_wrong"], s["net_drift_cents"], s["abs_drift_cents"], s["events_lost"],
            s["applied_twice"], s["profiles_wrong"], r["stats"]["redelivered"])


# ------------------------------------------------------------------ the stream

def test_the_stream_has_the_planted_sizes(full_env):
    p = full_env.key["planted"]
    assert (p["wallet_events"], p["profile_events"], p["distinct_events"]) == (45000, 5000, 50000)
    assert p["log_records"] == 51000 and p["retried_events"] == 1000
    assert (p["retried_wallet_events"], p["retried_profile_events"]) == (891, 109)
    assert p["late_profile_updates"] == 400 and p["out_of_order_arrivals"] == 400
    assert p["tie_accounts"] == 40 and p["accounts_with_profile"] == 1831
    assert full_env.manifest["partition_lengths"] == [16638, 16771, 17591]
    assert len(full_env.crashes) == 52


# ------------------------------------------------------------------ strategy scoreboard

def test_scoreboard_numbers(scoreboard):
    for name, expected in SCOREBOARD.items():
        assert row(scoreboard[name]) == expected, name


def test_naive_is_wrong_from_producer_retries_alone(scoreboard):
    """No crashes at all, and 891 wallet retries are already 891 doubled events."""
    s = scoreboard["naive (no crashes)"]["score"]
    assert s["applied_twice"] == 891 and s["events_lost"] == 0 and s["net_drift_cents"] > 0


def test_crash_redelivery_adds_to_the_naive_error(scoreboard):
    clean, crashed = scoreboard["naive (no crashes)"]["score"], scoreboard["naive"]["score"]
    assert crashed["applied_twice"] > clean["applied_twice"]
    assert crashed["abs_drift_cents"] > clean["abs_drift_cents"]


def test_applied_twice_counts_extra_applications(scoreboard):
    """The README defines applied twice as extra applications, so an event applied
    three times counts twice. Naive is the only strategy where that difference shows."""
    multi = [n for _, n in scoreboard["naive"]["score"]["anomalies"] if n > 1]
    assert len(multi) == 3023 and sum(n - 1 for n in multi) == 3109
    assert sorted(set(multi)) == [2, 3, 4]
    for label in ["commit_first", "dedup_separate_txn", "atomic_offset_only"]:
        assert {n for _, n in scoreboard[label]["score"]["anomalies"] if n > 1} == {2}


def test_commit_first_loses_money(scoreboard):
    s = scoreboard["commit_first"]["score"]
    assert s["events_lost"] == 2442 and s["net_drift_cents"] < 0
    assert scoreboard["commit_first"]["stats"]["redelivered"] == 0


def test_dedup_in_a_separate_transaction_still_corrupts_on_crash(scoreboard):
    dedup, naive = scoreboard["dedup_separate_txn"]["score"], scoreboard["naive"]["score"]
    assert 0 < dedup["applied_twice"] < naive["applied_twice"]
    assert dedup["accounts_wrong"] > 0 and dedup["events_lost"] == 0


def test_atomic_has_zero_drift_and_still_redelivers(scoreboard):
    r = scoreboard["atomic"]
    assert r["score"]["accounts_wrong"] == 0 and r["score"]["abs_drift_cents"] == 0
    assert r["stats"]["crashes"] == 52 and r["stats"]["redelivered"] == 2470


def test_offsets_alone_do_not_remove_producer_retries(scoreboard):
    """Atomic offsets fix the crash error. The 891 retried wallet events remain."""
    assert row(scoreboard["atomic_offset_only"])[:5] == row(scoreboard["naive (no crashes)"])[:5]


def test_profile_policies(scoreboard):
    wrong = {k: scoreboard[k]["score"]["profiles_wrong"] for k in
             ("atomic_last_arrived_wins", "atomic_time_only_guard", "atomic")}
    assert wrong == {"atomic_last_arrived_wins": 353, "atomic_time_only_guard": 34, "atomic": 0}


def test_every_profile_policy_leaves_balances_exact_on_the_atomic_sink(scoreboard):
    for k in ("atomic_last_arrived_wins", "atomic_time_only_guard", "atomic"):
        assert scoreboard[k]["score"]["accounts_wrong"] == 0


def _wrong_profiles(full_env, strategy):
    path = ARTIFACTS_DIR / "sinks" / f"{strategy}_profiles.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        sink = {int(r["account_id"]): (r["tier"], int(r["daily_limit_cents"])) for r in csv.DictReader(fh)}
    return {int(a) for a, p in full_env.key["profiles"].items()
            if sink.get(int(a)) != (p["tier"], p["daily_limit_cents"])}


def test_last_arrived_wins_is_wrong_only_where_an_update_arrived_late(full_env, scoreboard):
    newest, late = {}, set()
    for r in full_env.stream.records:
        if r["type"] == "profile_updated":
            a = r["account_id"]
            if r["version"] < newest.get(a, 0) and r["producer_attempt"] == 0:
                late.add(a)
            newest[a] = max(newest.get(a, 0), r["version"])
    wrong = _wrong_profiles(full_env, "atomic_last_arrived_wins")
    # one late account is rescued when a producer retry of its newest update lands after the late one
    assert len(late) == 354 and len(wrong) == 353 and wrong <= late


def test_time_only_guard_is_wrong_only_on_tied_accounts(full_env, scoreboard):
    by_account = {}
    for r in full_env.stream.records:
        if r["type"] == "profile_updated" and r["producer_attempt"] == 0:
            by_account.setdefault(r["account_id"], []).append(r["event_time"])
    tied = {a for a, times in by_account.items() if len(set(times)) < len(times)}
    wrong = _wrong_profiles(full_env, "atomic_time_only_guard")
    assert len(tied) == 40 and len(wrong) == 34 and wrong <= tied


# ------------------------------------------------------------------ the replay proof

def test_replay_atomic_is_checksum_identical_and_still_exact(replays):
    p = replays["atomic"]
    assert p["identical"] and (p["balances_differ"], p["profiles_differ"], p["processed_differ"]) == (0, 0, 0)
    assert p["wrong_before"] == p["wrong_after"] == 0


def test_replay_naive_and_commit_first_change_the_sink(replays):
    for name in ("naive", "commit_first"):
        p = replays[name]
        assert not p["identical"] and p["balances_differ"] == 2000, name
        assert p["net_drift_after"] > p["net_drift_before"], name
    assert replays["naive"]["wrong_before"] == 1580 and replays["naive"]["wrong_after"] == 2000


def test_replay_state_events_are_safe_to_replay_even_when_naive(replays):
    """An overwrite is idempotent. That is why only the delta table moves for naive."""
    assert replays["naive"]["profiles_differ"] == 0


def test_replay_dedup_is_idempotent_but_not_correct(replays):
    """The replay proof detects non-idempotence. It cannot repair earlier damage."""
    p = replays["dedup_separate_txn"]
    assert p["identical"] and p["wrong_before"] == p["wrong_after"] == 1355


# ------------------------------------------------------------------ batch size vs redelivery

def test_batch_size_changes_redelivery_and_never_correctness(batch_runs):
    expected = {50: (1020, 52419, 1419), 100: (510, 53470, 2470),
                500: (102, 62957, 11957), 5000: (11, 174952, 123952)}
    for b, (txns, handled, redelivered) in expected.items():
        r = batch_runs[b]
        assert r["stats"]["polls"] - r["stats"]["crashes"] == txns, b
        assert (r["stats"]["records_handled"], r["stats"]["redelivered"]) == (handled, redelivered), b
        s = r["score"]
        assert (s["accounts_wrong"], s["profiles_wrong"], s["abs_drift_cents"]) == (0, 0, 0), b
    redelivered = [batch_runs[b]["stats"]["redelivered"] for b in sorted(batch_runs)]
    assert redelivered == sorted(redelivered)


# ------------------------------------------------------------------ the worked account

def _anomalies(scoreboard, name, event_ids):
    return {e: n for e, n in scoreboard[name]["score"]["anomalies"] if e in event_ids}


def test_featured_account_shows_a_duplicate_a_crash_and_an_out_of_order_update(full_env, scoreboard):
    f = full_env.manifest["featured"]
    assert (f["account_id"], f["partition"], f["crash_offset"], f["crash_event_id"]) == (172, 0, 304, "w-000953")
    mine = {r["event_id"] for r in full_env.stream.records if r["account_id"] == 172}
    assert _anomalies(scoreboard, "naive", mine) == {"w-000233": 2, "w-000953": 2}
    assert _anomalies(scoreboard, "commit_first", mine) == {"w-000233": 2, "w-000953": 0}
    assert _anomalies(scoreboard, "dedup_separate_txn", mine) == {"w-000953": 2}
    assert _anomalies(scoreboard, "atomic", mine) == {}


def test_inspect_prints_the_trace_in_plain_ascii(scoreboard):
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["inspect"]) == 0
    out = buf.getvalue()
    assert out.isascii()
    for marker in ("DUPLICATE (producer retry)", "OUT OF ORDER", "CONSUMER CRASH HERE", "w-000233"):
        assert marker in out


# ------------------------------------------------------------------ determinism and output

def test_running_a_strategy_twice_gives_identical_numbers(full_env, scoreboard):
    from log_replay.experiments import make_job, run_jobs
    again = run_jobs([make_job(full_env.cfg, "dedup_separate_txn", full_env.crashes)], workers=1)[0]
    assert row(again) == row(scoreboard["dedup_separate_txn"])


def test_sink_exports_are_ascii_and_sorted(scoreboard):
    path = ARTIFACTS_DIR / "sinks" / "atomic_balances.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "account_id,balance_cents" and len(lines) == 2001
    ids = [int(x.split(",")[0]) for x in lines[1:]]
    assert ids == sorted(ids)


def test_notebook_numbers_match_the_scripts(scoreboard):
    """The executed notebook prints one JSON line of headline numbers. They must equal run.py's."""
    from pathlib import Path
    import pytest
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "event_log_replay_standalone.ipynb"
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    found = None
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            for line in text.splitlines():
                if line.startswith("NOTEBOOK_RESULTS "):
                    found = json.loads(line[len("NOTEBOOK_RESULTS "):])
    assert found, "notebook has no NOTEBOOK_RESULTS line, execute it first"
    for name, expected in SCOREBOARD.items():
        assert tuple(found["scoreboard"][name]) == expected, name
