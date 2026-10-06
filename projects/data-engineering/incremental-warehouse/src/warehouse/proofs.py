"""The three properties, each as a function that returns what it measured.

    rerun      run a date twice, nothing changes
    backfill   run past dates late, each partition holds its own data
    break      bad input fails before publish, the last good tables stay

None of them reads the answer key. They compare the pipeline with itself, which
is the point. Properties of an operation should not need an oracle.
"""
from datetime import date

from warehouse import extract
from warehouse.clock import FrozenClock
from warehouse.harness import Scenario
from warehouse.naive import day_number


def clean_checksums(cfg, days, opts, upto):
    """Published checksums of a plain in-memory pass through day `upto`."""
    sc = Scenario(cfg, days, opts)
    for day in range(1, upto + 1):
        sc.run_day(day)
    return sc.wh.checksums()


def rerun(cfg, days, opts, root, iso_date):
    day = day_number(cfg, iso_date)
    sc = Scenario(cfg, days, opts, root=root)
    for d in range(1, day):
        sc.run_day(d)
    first = sc.run_day(day)
    snap1 = {"checksums": sc.wh.checksums(), "published_revenue": sc.wh.published_revenue(),
             "fct_rows": sc.wh.conn.execute(
                 "SELECT count(*) FROM published.fct_order").fetchone()[0],
             "raw_orders_digest": sc.store.digest("orders", sc.date_of(day))}
    second = sc.run_day(day)
    snap2 = {"checksums": sc.wh.checksums(), "published_revenue": sc.wh.published_revenue(),
             "fct_rows": sc.wh.conn.execute(
                 "SELECT count(*) FROM published.fct_order").fetchone()[0],
             "raw_orders_digest": sc.store.digest("orders", sc.date_of(day))}
    return {"date": iso_date, "run_ids": [first.run_id, second.run_id],
            "both_ok": first.ok and second.ok, "first": snap1, "second": snap2,
            "identical": snap1 == snap2}


def backfill(cfg, days, opts, root, start_iso, end_iso):
    """Run history, skip a window, then fill it in with the wall clock far in the future."""
    start, end = day_number(cfg, start_iso), day_number(cfg, end_iso)
    last = cfg["source"]["days"]
    reference = clean_checksums(cfg, days, opts, last)
    today = date.fromisoformat(cfg["demo"]["backfill_wall_today"])
    sc = Scenario(cfg, days, opts, root=root, clock=FrozenClock(today))
    for d in range(1, start):
        sc.run_day(d)
    backfilled = [sc.run_day(d) for d in range(start, end + 1)]
    for d in range(end + 1, last + 1):
        sc.run_day(d)
    partitions = []
    for d in range(start, end + 1):
        part_date = sc.date_of(d)
        lo, hi = extract.window(part_date, opts.lookback_minutes)
        stamps = sc.store.read("orders", part_date)["updated_at"].to_pylist()
        partitions.append({"partition": part_date.isoformat(), "rows": len(stamps),
                           "min_updated_at": min(stamps), "max_updated_at": max(stamps),
                           "inside_own_window": all(lo <= s < hi for s in stamps)})
    final = sc.wh.checksums()
    return {"start": start_iso, "end": end_iso, "all_ok": all(r.ok for r in backfilled),
            "partitions": partitions,
            "partition_dates": [d.isoformat() for d in sc.store.partitions("orders")],
            "stray": [d.isoformat() for d in sc.store.partitions("orders") if d >= today],
            "reference": reference, "final": final, "equals_clean_pass": final == reference}


def break_demo(cfg, days, opts, root, kind, iso_date):
    day = day_number(cfg, iso_date)
    sc = Scenario(cfg, days, opts, root=root)
    for d in range(1, day):
        sc.run_day(d)
    before = {"revenue": sc.wh.published_revenue(), "checksums": sc.wh.checksums()}
    run = sc.run_day(day, break_kind=kind)
    after = {"revenue": sc.wh.published_revenue(), "checksums": sc.wh.checksums()}
    failed = sc.wh.conn.execute(
        "SELECT stage, check_name, detail FROM ops.check_results "
        "WHERE run_id = ? AND NOT passed ORDER BY stage, check_name", [run.run_id]).fetchall()
    broken = {"run_id": run.run_id, "run_ok": run.ok,
              "failed_task": run.failed_task().task if run.failed_task() else None,
              "skipped": run.skipped(), "failed_checks": failed,
              "published_revenue_before": before["revenue"],
              "published_revenue_after": after["revenue"],
              "published_unchanged": before["checksums"] == after["checksums"],
              "quarantined": sc.store.quarantined()}
    # The runbook path. The source is fixed (the replay rebuilds it clean), rerun the date.
    fixed = sc.run_day(day)
    reference = clean_checksums(cfg, days, opts, day)
    broken["rerun_after_fix_ok"] = fixed.ok
    broken["rerun_run_id"] = fixed.run_id
    broken["rerun_equals_clean_pass"] = sc.wh.checksums() == reference
    broken["published_revenue_after_fix"] = sc.wh.published_revenue()
    return broken
