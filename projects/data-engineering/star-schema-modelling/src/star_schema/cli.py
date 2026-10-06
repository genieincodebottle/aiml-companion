"""Command line entry point.

    python run.py data      generate the source CSVs and the answer key
    python run.py build     build the correct dbt model and run every test
    python run.py check     score the correct model against the answer key
    python run.py naive     build the naive version of each trap and score it
    python run.py break     the mutation matrix, 13 breaks against every test
    python run.py all       everything above, in order
"""
from __future__ import annotations

import argparse
import time

import duckdb

from star_schema import variants as V
from star_schema.config import load_config, resolve
from star_schema.data.io import write_world
from star_schema.evaluation.answer_key import load_key, score_connection, showcase_orders
from star_schema.pipelines import matrix as M
from star_schema.pipelines import report as R
from star_schema.utils.tables import money, render
from star_schema.verdict import TIERS

WORKERS = 4


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("star-schema", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, text in (("data", "generate the source CSVs and the answer key"),
                       ("build", "build the correct dbt model and run every test"),
                       ("check", "score the correct model against the answer key"),
                       ("naive", "build the naive version of each trap and score it"),
                       ("break", "run the mutation matrix"),
                       ("all", "run every step in order")):
        sub.add_parser(name, help=text)
    return p


def cmd_data(cfg: dict) -> None:
    hashes = write_world(cfg)
    ledger = load_key(cfg)[2]
    rows = [["customers", ledger["n_customers"]], ["customer change rows (CDC)", ledger["n_customer_change_rows"]],
            ["products", ledger["n_products"]], ["orders", ledger["n_orders"]],
            ["order lines (true)", ledger["n_lines"]], ["order line rows sent", ledger["n_line_rows_sent"]]]
    print("Source tables written to", resolve(cfg, "raw_dir"))
    print(render(rows, ["source", "rows"]))
    traps = [["1 relocation", "customers who moved once / twice",
              f"{ledger['customers_relocating_once']} / {ledger['customers_relocating_twice']}"],
             ["1 relocation", "orders placed in a region the customer later left",
              f"{ledger['relocated_orders']} ({money(ledger['relocated_revenue_cents'])})"],
             ["2 mixed grain", "shipping counted again on extra lines",
              money(ledger["shipping_excess_cents"])],
             ["3 re-sent delivery", "duplicate line rows sent",
              f"{ledger['resent_rows']} ({money(ledger['resent_revenue_cents'])})"],
             ["4 late dimension", "orders placed before the customer's first row",
              f"{ledger['late_orders']} ({money(ledger['late_revenue_cents'])})"],
             ["5 boundary", "orders stamped on a region change second",
              f"{ledger['boundary_orders']} ({money(ledger['boundary_revenue_cents'])})"]]
    print("\nPlanted trap sizes, from the generator's own ledger")
    print(render(traps, ["trap", "what was planted", "count (revenue)"], "lll"))
    print("\nsha256 of the raw files (identical on every run)")
    print(render([[k, v[:16]] for k, v in sorted(hashes.items())], ["file", "sha256 prefix"], "ll"))


def _baseline(cfg: dict) -> dict:
    return M.build_variant(None, cfg)


def cmd_build(cfg: dict) -> dict:
    start = time.time()
    r = _baseline(cfg)
    if not r["run_ok"]:
        raise SystemExit("dbt run failed: " + "; ".join(r["run_errors"]))
    print("Rows per model")
    print(render([[t, f"{n:,}"] for t, n in r["rows"].items()], ["model", "rows"]))
    passed = sum(1 for t in r["tests"] if t["status"] == "pass")
    by_tier = {tier: sum(1 for t in r["tests"] if t["tier"] == tier) for tier in TIERS}
    print(f"\ndbt tests {passed}/{len(r['tests'])} passed "
          f"(generic {by_tier['generic']}, singular {by_tier['singular']}, added {by_tier['added']})")
    for t in r["tests"]:
        if t["status"] != "pass":
            print(f"  [FAIL] {t['label']} ({t['failures']} rows)")
    R.write_json(resolve(cfg, "artifacts") / "build_summary.json",
                 {"rows": r["rows"], "tests": r["tests"]})
    print(f"[{'OK' if passed == len(r['tests']) else 'FAIL'}] build finished in {time.time() - start:.0f}s")
    return r


def cmd_check(cfg: dict) -> None:
    db = resolve(cfg, "artifacts") / "work" / "baseline" / "warehouse.duckdb"
    if not db.exists():
        cmd_build(cfg)
    con = duckdb.connect(str(db), read_only=True)
    score = score_connection(con, cfg)
    rows = []
    for name, s in score.items():
        rows.append([name, money(s["true_total"]), money(s["report_total"]), money(s["abs_error"]),
                     f"{s['cells_wrong']} of {s['cells_true']}", "yes" if s["exact"] else "NO"])
    print("Correct model against the answer key")
    print(render(rows, ["report", "true total", "model total", "abs error", "wrong cells", "exact"],
                 "lrrrrl"))
    sc = cfg["showcase"]
    print(f"\n{sc['name']} (customer {sc['customer_id']}) moved {sc['region_from']} -> "
          f"{sc['region_to']} at {sc['moved_at']}")
    truth = {o["order_id"]: o for o in showcase_orders(cfg)}
    sql = f"""
        select o.order_id, cast(o.order_ts as varchar), cast(sum(l.line_revenue) as varchar),
               d.region, cur.region
        from fct_order o
        join dim_customer d on d.customer_sk = o.customer_sk
        join dim_customer cur on cur.customer_id = d.customer_id and cur.is_current
        join fct_order_line l on l.order_id = o.order_id
        where d.customer_id = {int(sc['customer_id'])}
        group by 1, 2, 4, 5 order by 1"""
    rows = []
    for order_id, ts, amount, model_region, current_region in con.execute(sql).fetchall():
        true_region = truth[order_id]["true_region"]
        rows.append([order_id, ts, amount, true_region, model_region, current_region,
                     "ok" if model_region == true_region else "WRONG",
                     "ok" if current_region == true_region else "WRONG"])
    print(render(rows, ["order_id", "order_ts", "amount", "true region", "model region",
                        "is_current region", "model", "is_current join"], "lllllllr"))
    con.close()


def cmd_naive(cfg: dict) -> None:
    start = time.time()
    results = M.build_many(V.NAIVE_IDS, cfg, run_tests=False, workers=WORKERS)
    print("Naive approach against the answer key (the correct model is exact, see `check`)")
    R.print_naive(results)
    R.write_naive_artifacts(results, resolve(cfg, "artifacts"))
    print(f"[OK] {len(results)} naive builds in {time.time() - start:.0f}s")


def cmd_break(cfg: dict) -> list[dict]:
    start = time.time()
    results = M.build_many(R.variant_ids(), cfg, run_tests=True, workers=WORKERS)
    broken = [r for r in results if not r["run_ok"]]
    if broken:
        raise SystemExit("dbt run failed for: " + ", ".join(r["id"] for r in broken))
    print("Mutation matrix. Every mutation is a plausible edit to the shipped model.")
    R.print_matrix(results)
    R.write_matrix_artifacts(results, resolve(cfg, "artifacts"))
    print(f"[OK] {len(results)} builds in {time.time() - start:.0f}s")
    return results


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    steps = {"data": cmd_data, "build": cmd_build, "check": cmd_check,
             "naive": cmd_naive, "break": cmd_break}
    if args.cmd == "all":
        for name in ("data", "build", "check", "naive", "break"):
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            steps[name](cfg)
    else:
        steps[args.cmd](cfg)
    return 0
