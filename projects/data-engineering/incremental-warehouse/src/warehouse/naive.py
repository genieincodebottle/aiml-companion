"""Each naive variant against the correct pipeline, measured on the same shop.

A variant is the correct pipeline with one switch changed (see `options.py`).
Naive runs use `checks_mode = report`, so the checks log failures but the run
carries on and publishes, which is what a pipeline with a separate report does.
Every number comes from the answer key through `scoring.py`.
"""
from datetime import date

from warehouse import scoring
from warehouse.clock import FrozenClock
from warehouse.harness import Scenario


def _pass(cfg, days, opts, upto=None, clock=None):
    sc = Scenario(cfg, days, opts, clock=clock)
    for day in range(1, (upto or cfg["source"]["days"]) + 1):
        sc.run_day(day)
    return sc


def day_number(cfg, iso_date):
    return (date.fromisoformat(iso_date) - date.fromisoformat(cfg["source"]["start_date"])).days + 1


def first_failure(wh, stage):
    """(run_date, check_name) of the earliest failed check in a stage, or None."""
    row = wh.conn.execute(
        "SELECT CAST(run_date AS VARCHAR), check_name FROM ops.check_results "
        "WHERE stage = ? AND NOT passed ORDER BY run_date, check_name LIMIT 1",
        [stage]).fetchone()
    return tuple(row) if row else None


def _final(key, sc, last_day):
    return {
        "orders": scoring.score_orders(key, sc.wh, last_day),
        "revenue": scoring.score_revenue(key, sc.wh.revenue_by_date_region(), last_day),
        "every_day": scoring.score_every_day(key, sc.published_by_day),
        "first_marts_failure": first_failure(sc.wh, "check_marts"),
    }


def first_stopped_run(cfg, days, opts):
    """Run days in order with gating checks until a run stops. Returns what stopped it."""
    sc = Scenario(cfg, days, opts)
    for day in range(1, cfg["source"]["days"] + 1):
        run = sc.run_day(day)
        if not run.ok:
            failed = sc.wh.conn.execute(
                "SELECT check_name FROM ops.check_results WHERE run_id = ? AND NOT passed "
                "ORDER BY check_name", [run.run_id]).fetchall()
            return {"day": day, "failed_task": run.failed_task().task,
                    "failed_checks": [name for (name,) in failed],
                    "published_unchanged": sc.published_by_day[day] == sc.published_by_day.get(day - 1, [])}
    return None


def late_commits(cfg, days, key, correct, last_day):
    strict = _pass(cfg, days, correct.opts.but(lookback_minutes=0, checks_mode="report"))
    out = {name: {**_final(key, sc, last_day),
                  "late": scoring.late_commit_coverage(key, sc.wh, sc.store)}
           for name, sc in (("strict", strict), ("lookback", correct))}
    out["strict_gated"] = first_stopped_run(cfg, days, correct.opts.but(lookback_minutes=0))
    return out


def hard_deletes(cfg, days, key, correct, last_day):
    naive = _pass(cfg, days, correct.opts.but(detect_deletes=False, checks_mode="report"))
    return {name: _final(key, sc, last_day)
            for name, sc in (("no_detection", naive), ("reconciled", correct))}


def _fact_state(wh):
    rows, revenue = wh.conn.execute(
        "SELECT count(*), coalesce(sum(revenue_cents), 0) FROM published.fct_order").fetchone()
    return {"rows": rows, "revenue": int(revenue)}


def append_vs_overwrite(cfg, days, key, correct, last_day):
    """Run one date twice, then finish the month, for each load mode."""
    rerun_day = day_number(cfg, cfg["demo"]["rerun_date"])
    out = {}
    for mode in ("append", "overwrite"):
        sc = Scenario(cfg, days, correct.opts.but(load_mode=mode, checks_mode="report"))
        for day in range(1, rerun_day + 1):
            sc.run_day(day)
        before = _fact_state(sc.wh)
        sc.run_day(rerun_day)
        after = _fact_state(sc.wh)
        for day in range(rerun_day + 1, last_day + 1):
            sc.run_day(day)
        out[mode] = {
            "rerun_day": rerun_day,
            "rows_added_by_rerun": after["rows"] - before["rows"],
            "revenue_added_by_rerun": after["revenue"] - before["revenue"],
            **_final(key, sc, last_day),
        }
    return out


def wall_clock(cfg, days, key, correct, last_day):
    """Backfill five past days with a task that files data under today's date."""
    start = day_number(cfg, cfg["demo"]["backfill_start"])
    end = day_number(cfg, cfg["demo"]["backfill_end"])
    today = date.fromisoformat(cfg["demo"]["backfill_wall_today"])
    opts = correct.opts.but(checks_mode="report")
    sc = Scenario(cfg, days, opts, clock=FrozenClock(today))
    after_backfill = {}
    for day in range(1, last_day + 1):
        sc.set_options(opts.but(clock="wall") if start <= day <= end else opts)
        sc.run_day(day)
        if day == end:
            after_backfill = {
                "orders": scoring.score_orders(key, sc.wh, end),
                "revenue": scoring.score_revenue(key, sc.wh.revenue_by_date_region(), end)}
    backfilled = [sc.date_of(d) for d in range(start, end + 1)]
    present = sc.store.partitions("orders")
    return {
        "days_backfilled": len(backfilled),
        "partitions_present": sum(1 for d in backfilled if d in present),
        "stray_partition": today.isoformat(),
        "rows_in_stray_partition": sc.store.row_counts("orders").get(today, 0),
        "after_backfill": after_backfill,
        **_final(key, sc, last_day),
    }


def breaks(cfg, days, key, correct):
    """Inject each break on one day, with checks as a report and as a gate."""
    day = day_number(cfg, cfg["demo"]["break_date"])
    out = {}
    for kind in ("rename", "nulls", "volume"):
        out[kind] = {}
        for mode in ("report", "gate"):
            sc = Scenario(cfg, days, correct.opts.but(checks_mode=mode))
            for d in range(1, day):
                sc.run_day(d)
            before = sc.wh.published_revenue()
            sums_before = sc.wh.checksums()
            run = sc.run_day(day, break_kind=kind)
            rows = sc.wh.revenue_by_date_region()
            score = scoring.score_revenue(key, rows, day)
            stale = scoring.score_revenue(key, rows, day - 1)
            fail = first_failure(sc.wh, "check_raw") or first_failure(sc.wh, "check_marts")
            failed_task = run.failed_task()
            out[kind][mode] = {
                "day": day,
                "run_ok": run.ok,
                "published_before": before,
                "published_after": sc.wh.published_revenue(),
                "truth_after": score["truth_cents"],
                "error_cents": score["abs_error_cents"],
                "error_vs_day_before_cents": stale["abs_error_cents"],
                "no_region_cents": sum(c for _, region, c in sc.wh.revenue_by_date_region()
                                       if region is None),
                "first_failed_check": fail[1] if fail else None,
                "failed_task": failed_task.task if failed_task else None,
                "skipped": run.skipped(),
                "published_unchanged": sc.wh.checksums() == sums_before,
            }
    return out


def scd_type(cfg, days, key, correct, last_day):
    naive = _pass(cfg, days, correct.opts.but(scd_type=1, checks_mode="report"))
    return {name: _final(key, sc, last_day)
            for name, sc in (("type1", naive), ("type2", correct))}


def run_all(cfg, days, key, correct, log=None):
    """Every comparison. `correct` is a finished clean Scenario."""
    last = cfg["source"]["days"]
    steps = [("late_commits", late_commits), ("hard_deletes", hard_deletes),
             ("append_vs_overwrite", append_vs_overwrite), ("wall_clock", wall_clock),
             ("scd_type", scd_type)]
    out = {}
    for name, fn in steps:
        out[name] = fn(cfg, days, key, correct, last)
        if log:
            log(name)
    out["breaks"] = breaks(cfg, days, key, correct)
    if log:
        log("breaks")
    return out
