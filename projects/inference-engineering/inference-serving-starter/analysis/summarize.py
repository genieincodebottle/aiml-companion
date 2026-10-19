"""Print the result table for one sweep file, and optionally the knee plot. GIVEN glue
over the functions YOU wrote.

    python -m analysis.summarize results/l4_bf16.json
    python -m analysis.summarize results/l4_bf16.json --plot results/knee_l4_bf16.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.goodput import goodput, max_slo_concurrency, slo_attainment
from analysis.results_io import load_results
from analysis.knee import find_knee, plot_knee
from analysis.tables import latency_table, throughput
from bench.records import drop_warmup


def summarize(path) -> dict:
    meta, levels = load_results(path)
    slo = meta["slo"]
    rows = []
    steady_levels = []
    for lv in levels:
        recs = drop_warmup(lv["records"], warmup_requests=meta.get("warmup_requests", 0))
        steady_levels.append({"concurrency": lv.get("concurrency"), "records": recs})
        lat = latency_table(recs)
        thr = throughput(recs)
        rows.append({
            "concurrency": lv.get("concurrency"), "qps": lv.get("qps"), "n_ok": lat["n_ok"], "n_failed": lat["n_failed"],
            "output_tok_per_s": thr["output_tok_per_s"], "req_per_s": thr["req_per_s"],
            "goodput_req_per_s": goodput(recs, slo["ttft_s"], slo["tpot_s"]),
            "slo_attainment": slo_attainment(recs, slo["ttft_s"], slo["tpot_s"]),
            "p50_ttft_s": lat["ttft_s"].get(50), "p99_ttft_s": lat["ttft_s"].get(99),
            "p50_tpot_s": lat["tpot_s"].get(50), "p99_itl_s": lat["itl_s"].get(99),
            "p50_e2e_s": lat["e2e_s"].get(50),
        })
    pts = [(r["concurrency"], r["output_tok_per_s"]) for r in rows if r["concurrency"] is not None]
    knee = find_knee(pts) if pts else None
    best = max_slo_concurrency(steady_levels, slo["ttft_s"], slo["tpot_s"]) if pts else None
    return {"name": meta["name"], "slo": slo, "rows": rows, "knee_concurrency": knee, "max_slo_concurrency": best}


def print_table(s: dict) -> None:
    print(f"{s['name']}   SLO: TTFT <= {s['slo']['ttft_s']} s, TPOT <= {s['slo']['tpot_s']} s")
    print(f"{'conc':>5} {'out tok/s':>10} {'goodput':>8} {'attain':>7} {'p50 ttft':>9} {'p99 ttft':>9} {'p99 itl':>8} {'fail':>5}")
    for r in s["rows"]:
        f = lambda v: "      n/a" if v is None else f"{v:9.3f}"
        print(f"{r['concurrency'] if r['concurrency'] is not None else r['qps']:>5} {r['output_tok_per_s']:10.1f} "
              f"{r['goodput_req_per_s']:8.2f} {r['slo_attainment']:7.2f} {f(r['p50_ttft_s'])} {f(r['p99_ttft_s'])} "
              f"{(r['p99_itl_s'] or 0):8.3f} {r['n_failed']:5d}")
    print(f"knee at concurrency {s['knee_concurrency']}; highest concurrency meeting the SLO: {s['max_slo_concurrency']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None, help="also write the summary here")
    a = ap.parse_args(argv)
    s = summarize(a.result)
    print_table(s)
    if a.plot:
        pts = [{"concurrency": r["concurrency"], "output_tok_per_s": r["output_tok_per_s"],
                "p99_ttft_s": r["p99_ttft_s"] or 0.0, "p99_itl_s": r["p99_itl_s"] or 0.0} for r in s["rows"]]
        plot_knee(pts, a.plot, knee=s["knee_concurrency"], title=s["name"])
        print(f"wrote {a.plot}")
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
