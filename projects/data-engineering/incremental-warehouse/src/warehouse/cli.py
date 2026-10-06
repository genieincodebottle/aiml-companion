"""Command line. Every number the README quotes is printed by one of these commands."""
import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter

from warehouse import naive, proofs, scoring
from warehouse.config import ARTIFACTS, DATA, load_config
from warehouse.harness import Scenario
from warehouse.options import PipelineOptions
from warehouse.source.shop import Shop
from warehouse.source.world import generate_world


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def usd(cents):
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def table(headers, rows, title=None):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [title] if title else []
    out += [line, "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |", line]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    out.append(line)
    print("\n".join(out))


def short(digest):
    return digest[:12]


def save_json(name, obj):
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                                  encoding="utf-8", newline="\n")


def save_csv(name, headers, rows):
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / name, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(headers)
        w.writerows(rows)


def status_line(ok, text):
    print(f"[{'OK' if ok else 'FAIL'}] {text}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_source(cfg, args):
    days, key = generate_world(cfg)
    shop = Shop(":memory:", days)
    rows, volume = [], []
    for batch in days:
        day = batch["day"]
        shop.advance_to(day)
        ops = Counter(e["op"] for e in batch["events"])
        late = sum(1 for e in batch["events"] if e["late"])
        counts = shop.counts()
        rows.append([day, batch["date"], len(batch["events"]), ops["insert_order"],
                     ops["update_order"], ops["delete_order"], late,
                     counts["customers"], counts["orders"], counts["order_lines"]])
        volume.append(rows[-1])
    table(["day", "date", "events", "new orders", "order updates", "deletes", "late",
           "customers", "orders", "order_lines"], rows,
          "Source database, activity per day and rows at the end of the day")
    planted = scoring.planted_summary(key)
    total = shop.counts()
    print()
    table(["measure", "value"], [
        ["days simulated", len(days)],
        ["customers at the end", total["customers"]],
        ["orders at the end", total["orders"]],
        ["order_lines at the end", total["order_lines"]],
        ["orders ever created", planted["orders_total"]],
        ["late commits planted", planted["late_commits"]],
        ["  on orders", planted["late_commits_orders"]],
        ["  on customers", planted["late_commits_customers"]],
        ["hard deletes planted", planted["hard_deletes"]],
    ], "Planted failures")
    save_csv("source_volume.csv", ["day", "date", "events", "new_orders", "order_updates",
                                   "deletes", "late", "customers", "orders", "order_lines"],
             volume)
    save_json("source_summary.json", {**planted, **{f"end_{k}": v for k, v in total.items()}})


def _daily(cfg, days, opts, root):
    sc = Scenario(cfg, days, opts, root=root, sleep=time.sleep)
    demo = cfg["demo"]
    rows = []
    for day in range(1, cfg["source"]["days"] + 1):
        iso = sc.date_of(day).isoformat()
        faults = ({demo["transient_task"]: demo["transient_failures"]}
                  if iso == demo["transient_date"] else None)
        run = sc.run_day(day, faults=faults)
        landed = run.xcom["land_raw"]["rows"]
        fct = sc.wh.conn.execute("SELECT count(*) FROM published.fct_order").fetchone()[0]
        attempts = [r for r in run.results if r.attempts > 1]
        rows.append([day, iso, "ok" if run.ok else "FAILED", landed["orders"],
                     len(run.xcom["plan_partitions"]), fct,
                     usd(sc.wh.published_revenue()),
                     ",".join(f"{t}x{n}" for t, n in [(r.task, r.attempts) for r in attempts])])
    table(["day", "date", "run", "orders landed", "partitions rebuilt",
           "fct_order rows", "published revenue", "retries"], rows,
          "One DAG run per logical date")
    return sc, rows


def cmd_daily(cfg, args):
    days, _ = generate_world(cfg)
    opts = PipelineOptions.from_cfg(cfg)
    sc, rows = _daily(cfg, days, opts, DATA / "clean")
    sums = sc.wh.checksums()
    print()
    table(["table", "checksum"], [[t, short(c)] for t, c in sums.items()],
          "Published tables after day 30")
    save_csv("daily.csv", ["day", "date", "run", "orders_landed", "partitions_rebuilt",
                           "fct_order_rows", "published_revenue", "retries"], rows)
    save_json("published_by_day.json", {str(d): rws for d, rws in sc.published_by_day.items()})
    save_json("daily_checksums.json", sums)
    status_line(all(r[2] == "ok" for r in rows), f"{len(rows)} daily runs, "
                f"{rows[-1][5]} fct_order rows, published {rows[-1][6]}")
    sc.close()


def _published_by_day():
    path = ARTIFACTS / "published_by_day.json"
    if not path.exists() or not (DATA / "clean" / "warehouse.duckdb").exists():
        return None
    return {int(d): [tuple(r) for r in rows]
            for d, rows in json.loads(path.read_text(encoding="utf-8")).items()}


def cmd_score(cfg, args):
    days, key = generate_world(cfg)
    published = _published_by_day()
    if published is None:
        print("No clean run found, running `daily` first.\n")
        cmd_daily(cfg, args)
        published = _published_by_day()
    rows, exact_days = [], 0
    for day in sorted(published):
        s = scoring.score_revenue(key, published[day], day)
        exact_days += s["exact"]
        rows.append([day, usd(s["truth_cents"]), usd(s["published_cents"]),
                     s["cells"], s["cells_wrong"], "match" if s["exact"] else "DIFFERS"])
    table(["day", "true revenue", "published", "date x region cells", "cells wrong", "verdict"],
          rows, "Published revenue against the answer key, after each daily run")
    from warehouse.warehouse import Warehouse
    wh = Warehouse(DATA / "clean" / "warehouse.duckdb")
    last = cfg["source"]["days"]
    orders = scoring.score_orders(key, wh, last)
    dim = scoring.score_dimension(key, wh, last)
    print()
    table(["measure", "value"], [
        ["true orders after day 30", orders["orders_true"]],
        ["published fct_order rows", orders["fct_rows"]],
        ["orders missing", orders["orders_missing"]],
        ["hard-deleted orders still published", orders["deleted_still_published"]],
        ["orders with a stale status", orders["status_stale"]],
        ["orders in the wrong region", orders["region_wrong"]],
        ["true dimension rows", dim["truth_rows"]],
        ["published dimension rows", dim["published_rows"]],
        ["dimension equals the truth", dim["exact"]],
    ], "Order level and dimension level comparison after day 30")
    wh.conn.close()
    ok = exact_days == len(rows) and not any(
        orders[k] for k in ("orders_missing", "deleted_still_published", "status_stale",
                            "region_wrong", "duplicate_rows")) and dim["exact"]
    save_json("score.json", {"days_exact": exact_days, "days": len(rows),
                             "orders": orders, "dimension": dim, "exact": ok})
    status_line(ok, f"published revenue matches the answer key on {exact_days} of "
                f"{len(rows)} days, and the day-30 tables match order by order")


def cmd_rerun(cfg, args):
    days, _ = generate_world(cfg)
    iso = args.date or cfg["demo"]["rerun_date"]
    r = proofs.rerun(cfg, days, PipelineOptions.from_cfg(cfg), DATA / "rerun", iso)
    rows = [["run id", r["run_ids"][0], r["run_ids"][1]],
            ["fct_order rows", r["first"]["fct_rows"], r["second"]["fct_rows"]],
            ["published revenue", usd(r["first"]["published_revenue"]),
             usd(r["second"]["published_revenue"])],
            ["raw orders partition sha256", short(r["first"]["raw_orders_digest"]),
             short(r["second"]["raw_orders_digest"])]]
    rows += [[f"checksum {t}", short(c), short(r["second"]["checksums"][t])]
             for t, c in r["first"]["checksums"].items()]
    table(["", "first run", "second run"], rows, f"Run {iso} twice")
    save_json("rerun.json", r)
    status_line(r["identical"] and r["both_ok"],
                "rerun left every checksum and the raw partition file unchanged")


def cmd_backfill(cfg, args):
    days, _ = generate_world(cfg)
    start = args.start or cfg["demo"]["backfill_start"]
    end = args.end or cfg["demo"]["backfill_end"]
    r = proofs.backfill(cfg, days, PipelineOptions.from_cfg(cfg), DATA / "backfill", start, end)
    table(["partition", "orders rows", "oldest updated_at", "newest updated_at",
           "inside its own window"],
          [[p["partition"], p["rows"], p["min_updated_at"], p["max_updated_at"],
            p["inside_own_window"]] for p in r["partitions"]],
          f"Backfilled {start} to {end} with the wall clock at "
          f"{cfg['demo']['backfill_wall_today']}")
    print()
    table(["table", "clean single pass", "after backfill"],
          [[t, short(r["reference"][t]), short(r["final"][t])] for t in r["reference"]],
          "Final published tables")
    save_json("backfill.json", r)
    ok = r["equals_clean_pass"] and not r["stray"] and all(
        p["inside_own_window"] for p in r["partitions"])
    status_line(ok, f"{len(r['partition_dates'])} partitions, none dated by the wall clock, "
                "final warehouse equals the clean single pass")


def cmd_break(cfg, args):
    days, _ = generate_world(cfg)
    iso = args.date or cfg["demo"]["break_date"]
    kinds = ["rename", "nulls", "volume"] if args.kind == "all" else [args.kind]
    results = {}
    for kind in kinds:
        r = proofs.break_demo(cfg, days, PipelineOptions.from_cfg(cfg),
                              DATA / f"break_{kind}", kind, iso)
        results[kind] = r
        print(f"\nBreak '{kind}' injected on {iso}")
        table(["measure", "value"], [
            ["run ok", r["run_ok"]],
            ["failed task", r["failed_task"]],
            ["failed checks", "; ".join(f"{c[1]}" for c in r["failed_checks"])],
            ["tasks skipped", f"{len(r['skipped'])} ({', '.join(r['skipped'][:3])}, ...)"],
            ["published revenue before", usd(r["published_revenue_before"])],
            ["published revenue after", usd(r["published_revenue_after"])],
            ["published tables unchanged", r["published_unchanged"]],
            ["raw partitions quarantined", len(r["quarantined"])],
            ["rerun after fix ok", r["rerun_after_fix_ok"]],
            ["published revenue after fix", usd(r["published_revenue_after_fix"])],
            ["equals a clean pass to that day", r["rerun_equals_clean_pass"]],
        ])
        for stage, name, detail in r["failed_checks"]:
            print(f"    {stage}.{name}: {detail}")
        status_line(not r["run_ok"] and r["failed_task"] == "check_raw"
                    and r["published_unchanged"] and r["rerun_equals_clean_pass"],
                    f"{kind} stopped at {r['failed_task']}, publish skipped, "
                    "last good tables untouched, rerun after the fix is clean")
    save_json("break.json", results)


def cmd_naive(cfg, args):
    days, key = generate_world(cfg)
    opts = PipelineOptions.from_cfg(cfg)
    correct = Scenario(cfg, days, opts)
    for day in range(1, cfg["source"]["days"] + 1):
        correct.run_day(day)
    res = naive.run_all(cfg, days, key, correct,
                        log=lambda name: print(f"  measured {name}", flush=True))
    print()
    lc, hd, ao, wc, sd, br = (res[k] for k in (
        "late_commits", "hard_deletes", "append_vs_overwrite", "wall_clock", "scd_type",
        "breaks"))
    s, g = lc["strict"], lc["lookback"]
    table(["measure", "strict watermark", "lookback + dedup"], [
        ["late commits planted", s["late"]["planted"], g["late"]["planted"]],
        ["late commit rows landed", s["late"]["landed"], g["late"]["landed"]],
        ["late commit rows missed", s["late"]["missed"], g["late"]["missed"]],
        ["orders missing after day 30", s["orders"]["orders_missing"], g["orders"]["orders_missing"]],
        ["orders with a stale status", s["orders"]["status_stale"], g["orders"]["status_stale"]],
        ["orders in the wrong region", s["orders"]["region_wrong"], g["orders"]["region_wrong"]],
        ["days with exact revenue (of 30)", s["every_day"]["days_exact"], g["every_day"]["days_exact"]],
        ["revenue cells wrong on day 30", s["revenue"]["cells_wrong"], g["revenue"]["cells_wrong"]],
        ["abs revenue error on day 30", usd(s["revenue"]["abs_error_cents"]),
         usd(g["revenue"]["abs_error_cents"])],
        ["first reconcile failure", _fail(s), _fail(g)],
    ], "1. Late commits")
    sg = lc["strict_gated"]
    print(f"   With the strict extract and gating checks, day {sg['day']} stops at "
          f"{sg['failed_task']} ({', '.join(sg['failed_checks'])}) and publishes nothing.")
    n, c = hd["no_detection"], hd["reconciled"]
    print()
    table(["measure", "no delete detection", "key reconciliation"], [
        ["orders hard-deleted from the source", scoring.planted_summary(key)["hard_deletes"], "same"],
        ["deleted orders still published", n["orders"]["deleted_still_published"],
         c["orders"]["deleted_still_published"]],
        ["published revenue", usd(n["revenue"]["published_cents"]),
         usd(c["revenue"]["published_cents"])],
        ["true revenue", usd(n["revenue"]["truth_cents"]), usd(c["revenue"]["truth_cents"])],
        ["revenue overstated", usd(n["revenue"]["published_cents"] - n["revenue"]["truth_cents"]),
         usd(c["revenue"]["published_cents"] - c["revenue"]["truth_cents"])],
        ["first reconcile failure", _fail(n), _fail(c)],
    ], "2. Hard deletes")
    a, o = ao["append"], ao["overwrite"]
    print()
    table(["measure", "append", "partition overwrite"], [
        [f"rows added by running day {a['rerun_day']} twice", a["rows_added_by_rerun"],
         o["rows_added_by_rerun"]],
        ["revenue double counted by that rerun", usd(a["revenue_added_by_rerun"]),
         usd(o["revenue_added_by_rerun"])],
        ["fct_order rows after day 30", a["orders"]["fct_rows"], o["orders"]["fct_rows"]],
        ["true orders", a["orders"]["orders_true"], o["orders"]["orders_true"]],
        ["duplicate rows", a["orders"]["duplicate_rows"], o["orders"]["duplicate_rows"]],
        ["published revenue", usd(a["revenue"]["published_cents"]),
         usd(o["revenue"]["published_cents"])],
        ["true revenue", usd(a["revenue"]["truth_cents"]), usd(o["revenue"]["truth_cents"])],
        ["first grain failure", _fail(a), _fail(o)],
    ], "3. Append against partition overwrite")
    nb = wc["after_backfill"]
    print()
    table(["measure", "wall clock", "logical run_date"], [
        ["past days backfilled", wc["days_backfilled"], wc["days_backfilled"]],
        ["their own partitions present", wc["partitions_present"], wc["days_backfilled"]],
        [f"rows in stray partition {wc['stray_partition']}", wc["rows_in_stray_partition"], 0],
        ["orders missing right after the backfill", nb["orders"]["orders_missing"], 0],
        ["orders with a stale status right after", nb["orders"]["status_stale"], 0],
        ["abs revenue error right after", usd(nb["revenue"]["abs_error_cents"]), usd(0)],
        ["orders missing after day 30", wc["orders"]["orders_missing"], 0],
        ["published revenue after day 30", usd(wc["revenue"]["published_cents"]),
         usd(wc["revenue"]["truth_cents"])],
        ["first reconcile failure", _fail(wc), "none"],
    ], "4. Wall clock against logical run_date")
    print()
    rows = []
    for kind in ("rename", "nulls", "volume"):
        r, gt = br[kind]["report"], br[kind]["gate"]
        rows.append([kind, gt["first_failed_check"], usd(r["published_before"]),
                     usd(r["published_after"]), usd(r["error_cents"]),
                     usd(r["no_region_cents"]), usd(gt["published_after"]),
                     f"stops at {gt['failed_task']}", len(gt["skipped"])])
    table(["break", "check that fires", "published before", "published, report only",
           "abs error, report only", "revenue with no region", "published, gated",
           "gated run", "tasks skipped"],
          rows, "5. Checks as a report against checks as gating stages (break on day 20)")
    print(f"   True revenue on day 20 is {usd(br['rename']['gate']['truth_after'])}. "
          "The gated runs leave the day 19 numbers in place.")
    t1, t2 = sd["type1"], sd["type2"]
    print()
    table(["measure", "SCD type 1", "SCD type 2"], [
        ["orders in the wrong region", t1["orders"]["region_wrong"], t2["orders"]["region_wrong"]],
        ["date x region cells wrong", t1["revenue"]["cells_wrong"], t2["revenue"]["cells_wrong"]],
        ["abs revenue error across cells", usd(t1["revenue"]["abs_error_cents"]),
         usd(t2["revenue"]["abs_error_cents"])],
        ["total revenue error", usd(t1["revenue"]["published_cents"] - t1["revenue"]["truth_cents"]),
         usd(t2["revenue"]["published_cents"] - t2["revenue"]["truth_cents"])],
    ], "6. Current region against region at order time")
    save_json("naive.json", res)
    correct.close()
    status_line(True, "naive comparison written to artifacts/naive.json")


def _fail(block):
    f = block.get("first_marts_failure")
    return f"{f[0]} {f[1]}" if f else "none"


def cmd_runs(cfg, args):
    from warehouse.warehouse import Warehouse
    shown = False
    targets = ([(args.scenario, args.date)] if args.scenario else
               [("clean", cfg["demo"]["transient_date"]),
                *[(f"break_{k}", cfg["demo"]["break_date"]) for k in ("nulls",)]])
    for scenario, iso in targets:
        path = DATA / scenario / "warehouse.duckdb"
        if not path.exists():
            continue
        wh = Warehouse(path)
        ids = wh.conn.execute(
            "SELECT DISTINCT run_id FROM ops.runs WHERE run_date = ? "
            "ORDER BY CAST(split_part(run_id, '#', 2) AS INTEGER)", [iso]).fetchall()
        for (run_id,) in ids:
            table(["seq", "task", "status", "attempts", "error"],
                  [[r[1], r[2], r[3], r[4], r[5]] for r in wh.runs(run_id)],
                  f"ops.runs for {run_id} in scenario '{scenario}'")
            print()
            shown = True
        wh.conn.close()
    if not shown:
        print("No runs recorded yet. Run `python run.py daily` first.")


def cmd_all(cfg, args):
    shutil.rmtree(DATA, ignore_errors=True)
    args.date, args.start, args.end, args.kind, args.scenario = None, None, None, "all", None
    for name, fn in (("source", cmd_source), ("daily", cmd_daily), ("score", cmd_score),
                     ("rerun", cmd_rerun), ("backfill", cmd_backfill), ("break", cmd_break),
                     ("naive", cmd_naive), ("runs", cmd_runs)):
        print(f"\n===== {name} =====")
        fn(cfg, args)


COMMANDS = {"source": cmd_source, "daily": cmd_daily, "score": cmd_score,
            "rerun": cmd_rerun, "backfill": cmd_backfill, "break": cmd_break,
            "naive": cmd_naive, "runs": cmd_runs, "all": cmd_all}


def main(argv=None):
    p = argparse.ArgumentParser(prog="run.py", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="path to a config yaml")
    sub = p.add_subparsers(dest="command", required=True)
    h = {"source": "show the simulated source volume and the planted failures",
         "daily": "run all 30 days in order, with one transient failure",
         "score": "compare published revenue with the answer key",
         "rerun": "run one date twice, show nothing changes",
         "backfill": "run past dates late, show each partition holds its own data",
         "break": "inject a bad input, show the run fails before publish",
         "naive": "measure each naive variant against the correct pipeline",
         "runs": "print the ops.runs table for the retried and the broken run",
         "all": "run every command from a clean data folder"}
    for name, text in h.items():
        sp = sub.add_parser(name, help=text)
        if name in ("rerun", "break", "runs"):
            sp.add_argument("--date", default=None, help="logical date, YYYY-MM-DD")
        if name == "backfill":
            sp.add_argument("--start", default=None)
            sp.add_argument("--end", default=None)
        if name == "break":
            sp.add_argument("--kind", choices=["rename", "nulls", "volume", "all"],
                            default="all")
        if name == "runs":
            sp.add_argument("--scenario", default=None, help="folder under data/")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    COMMANDS[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
