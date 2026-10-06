"""Every claim the README makes, as an assertion.

The numbers are for seed 42 and the shipped conf/config.yaml. If a lesson stops
being true, or a refactor changes a number, this file goes red and the README is
wrong until someone fixes one of the two.
"""
from datetime import date

import pytest

from warehouse import proofs, scoring
from warehouse.harness import Scenario


# ---------------------------------------------------------------------------
# The correct pipeline matches the answer key
# ---------------------------------------------------------------------------
def test_published_revenue_matches_the_answer_key_after_every_day(key, clean):
    assert len(clean.published_by_day) == 30
    for day, rows in clean.published_by_day.items():
        s = scoring.score_revenue(key, rows, day)
        assert s["exact"], f"day {day} differs from the truth: {s}"
    final = scoring.score_revenue(key, clean.wh.revenue_by_date_region(), 30)
    assert final["published_cents"] == final["truth_cents"] == 51949481


def test_day_30_tables_match_the_truth_order_by_order(key, clean):
    o = scoring.score_orders(key, clean.wh, 30)
    assert o == {"orders_true": 1762, "fct_rows": 1762, "orders_missing": 0,
                 "deleted_still_published": 0, "status_stale": 0, "region_wrong": 0,
                 "duplicate_rows": 0}
    d = scoring.score_dimension(key, clean.wh, 30)
    assert d["exact"] and d["truth_rows"] == d["published_rows"] == 567


def test_the_planted_failures_are_all_caught_by_the_correct_pipeline(key, clean):
    cover = scoring.late_commit_coverage(key, clean.wh, clean.store)
    assert cover == {"planted": 111, "landed": 111, "missed": 0}
    assert scoring.planted_summary(key)["hard_deletes"] == 27


# ---------------------------------------------------------------------------
# 1. Late commits
# ---------------------------------------------------------------------------
def test_strict_watermark_misses_every_late_commit(results):
    s = results["late_commits"]["strict"]
    assert s["late"] == {"planted": 111, "landed": 0, "missed": 111}
    assert s["orders"]["orders_missing"] == 1
    assert s["orders"]["status_stale"] == 25
    assert s["orders"]["region_wrong"] == 9
    assert s["revenue"]["cells_wrong"] == 15
    assert s["revenue"]["abs_error_cents"] == 336616
    assert s["every_day"]["days_exact"] == 7
    assert tuple(s["first_marts_failure"]) == ("2025-06-04", "reconcile_by_date")


def test_lookback_with_dedup_loses_nothing(results):
    g = results["late_commits"]["lookback"]
    assert g["late"] == {"planted": 111, "landed": 111, "missed": 0}
    assert g["every_day"]["days_exact"] == 30
    assert g["revenue"]["abs_error_cents"] == 0
    assert g["first_marts_failure"] is None


def test_a_gated_strict_extract_stops_on_day_3_and_publishes_nothing(results):
    stopped = results["late_commits"]["strict_gated"]
    assert stopped == {"day": 3, "failed_task": "check_marts", "published_unchanged": True,
                       "failed_checks": ["reconcile_by_date", "reconcile_order_count",
                                         "reconcile_revenue"]}


# ---------------------------------------------------------------------------
# 2. Hard deletes
# ---------------------------------------------------------------------------
def test_without_delete_detection_deleted_orders_stay_in_revenue(results):
    n = results["hard_deletes"]["no_detection"]
    assert n["orders"]["deleted_still_published"] == 27
    assert n["revenue"]["published_cents"] - n["revenue"]["truth_cents"] == 774415
    assert tuple(n["first_marts_failure"]) == ("2025-06-05", "reconcile_by_date")
    c = results["hard_deletes"]["reconciled"]
    assert c["orders"]["deleted_still_published"] == 0 and c["revenue"]["exact"]


# ---------------------------------------------------------------------------
# 3. Append against partition overwrite
# ---------------------------------------------------------------------------
def test_append_double_counts_a_rerun_and_overwrite_does_not(results):
    a = results["append_vs_overwrite"]["append"]
    o = results["append_vs_overwrite"]["overwrite"]
    assert a["rerun_day"] == 15
    assert (a["rows_added_by_rerun"], a["revenue_added_by_rerun"]) == (249, 5807827)
    assert (o["rows_added_by_rerun"], o["revenue_added_by_rerun"]) == (0, 0)
    assert (a["orders"]["fct_rows"], a["orders"]["duplicate_rows"]) == (7027, 5238)
    assert a["revenue"]["published_cents"] == 164763743
    assert (o["orders"]["fct_rows"], o["orders"]["duplicate_rows"]) == (1762, 0)
    assert tuple(a["first_marts_failure"]) == ("2025-06-03", "grain_fct_order")


# ---------------------------------------------------------------------------
# 4. Wall clock against logical date
# ---------------------------------------------------------------------------
def test_wall_clock_partitions_lose_the_backfilled_days(results):
    w = results["wall_clock"]
    assert w["days_backfilled"] == 5 and w["partitions_present"] == 0
    assert (w["stray_partition"], w["rows_in_stray_partition"]) == ("2025-07-15", 235)
    assert w["after_backfill"]["orders"]["orders_missing"] == 114
    assert w["after_backfill"]["orders"]["status_stale"] == 230
    assert w["after_backfill"]["revenue"]["abs_error_cents"] == 4972437
    # The stray partition is also the newest key snapshot, so every order created
    # after day 13 is treated as deleted.
    assert w["orders"]["orders_missing"] == 1014
    assert w["revenue"]["published_cents"] == 22923404


# ---------------------------------------------------------------------------
# 5. Checks as a report against checks as gating stages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind, check, report_published", [
    ("rename", "contract_orders", 29136739),
    ("nulls", "nulls_orders", 34175794),
    ("volume", "volume", 32572262),
])
def test_a_report_publishes_bad_numbers_and_a_gate_does_not(results, kind, check, report_published):
    report, gate = results["breaks"][kind]["report"], results["breaks"][kind]["gate"]
    assert report["run_ok"] and report["published_after"] == report_published
    assert report["published_after"] != report["published_before"]
    assert not gate["run_ok"] and gate["failed_task"] == "check_raw"
    assert gate["first_failed_check"] == check
    assert gate["published_unchanged"] and gate["published_after"] == 32421301
    assert gate["error_vs_day_before_cents"] == 0     # the gated tables are exactly day 19
    assert report["error_vs_day_before_cents"] > 0
    assert len(gate["skipped"]) == 10 and gate["skipped"][-1] == "publish"


def test_each_break_names_its_cause_in_the_check_detail(cfg, days, opts, tmp_path):
    details = {}
    for kind in ("rename", "nulls", "volume"):
        r = proofs.break_demo(cfg, days, opts, None, kind, cfg["demo"]["break_date"])
        details[kind] = r["failed_checks"][0][2]
    assert details["rename"] == "missing=['status'] unexpected=['order_status']"
    assert details["nulls"] == "null rate above 0.01: {'customer_id': 0.344}"
    assert details["volume"] == "rows=55 trailing_median=248 ratio=0.22"


def test_the_nulls_report_files_revenue_under_no_region(results):
    assert results["breaks"]["nulls"]["report"]["no_region_cents"] == 1999811
    assert results["breaks"]["nulls"]["report"]["error_cents"] == 3999622


# ---------------------------------------------------------------------------
# 6. SCD type 1 against type 2
# ---------------------------------------------------------------------------
def test_scd_type_1_rewrites_history_into_the_wrong_region(results):
    t1, t2 = results["scd_type"]["type1"], results["scd_type"]["type2"]
    assert t1["orders"]["region_wrong"] == 218
    assert t1["revenue"]["cells_wrong"] == 99
    assert t1["revenue"]["abs_error_cents"] == 5552266
    assert t1["revenue"]["published_cents"] == t1["revenue"]["truth_cents"]  # moved, not lost
    assert t2["orders"]["region_wrong"] == 0 and t2["revenue"]["exact"]


# ---------------------------------------------------------------------------
# The three properties
# ---------------------------------------------------------------------------
def test_rerun_changes_nothing(cfg, days, opts, tmp_path):
    r = proofs.rerun(cfg, days, opts, tmp_path / "rerun", cfg["demo"]["rerun_date"])
    assert r["both_ok"] and r["identical"]
    assert r["run_ids"] == ["2025-06-16#1", "2025-06-16#2"]
    assert r["first"]["fct_rows"] == r["second"]["fct_rows"] == 904
    assert r["first"]["published_revenue"] == 25490291


def test_backfill_partitions_hold_their_own_dates_and_match_a_clean_pass(cfg, days, opts, tmp_path):
    d = cfg["demo"]
    r = proofs.backfill(cfg, days, opts, tmp_path / "bf", d["backfill_start"], d["backfill_end"])
    assert r["all_ok"] and r["equals_clean_pass"]
    assert len(r["partition_dates"]) == 30 and r["stray"] == []
    assert [p["rows"] for p in r["partitions"]] == [254, 236, 257, 268, 235]
    assert all(p["inside_own_window"] for p in r["partitions"])


@pytest.mark.parametrize("kind, check", [("rename", "contract_orders"),
                                         ("nulls", "nulls_orders"), ("volume", "volume")])
def test_a_break_fails_before_publish_and_the_fix_recovers(cfg, days, opts, tmp_path, kind, check):
    r = proofs.break_demo(cfg, days, opts, tmp_path / kind, kind, cfg["demo"]["break_date"])
    assert not r["run_ok"] and r["failed_task"] == "check_raw"
    assert [c[1] for c in r["failed_checks"]] == [check]
    assert len(r["skipped"]) == 10
    assert r["published_unchanged"]
    assert r["published_revenue_before"] == r["published_revenue_after"] == 32421301
    assert len(r["quarantined"]) == 4
    assert r["rerun_after_fix_ok"] and r["rerun_equals_clean_pass"]
    assert r["published_revenue_after_fix"] == 34175794


def test_the_demo_retry_shows_in_the_runs_table(cfg, days, opts):
    demo = cfg["demo"]
    sc = Scenario(cfg, days, opts)
    day = sc.day_of(date.fromisoformat(demo["transient_date"]))
    for earlier in range(1, day):
        sc.run_day(earlier)
    run = sc.run_day(day, faults={demo["transient_task"]: demo["transient_failures"]})
    rows = sc.wh.runs(run.run_id)
    extract = next(r for r in rows if r[2] == "extract_orders")
    assert extract[3:5] == ("success", 2)
    assert len(rows) == 14 and all(r[3] == "success" for r in rows)
