"""Turn one sweep into a cost and a fleet size and write results/decision.json. GIVEN glue
over the functions YOU wrote.

    python -m analysis.decide --sweep results/l4_fp8.json --gpu-hourly-usd 0.80 \
        --price-source "provider page URL, read 2026-10-20" --peak-qps 40

--gpu-hourly-usd is YOUR input. This repo has no price list, on purpose. The operating point
is the highest concurrency that still meets the SLO (analysis.goodput.max_slo_concurrency).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cost import cost_per_million_tokens
from analysis.fleet import fleet_plan
from analysis.results_io import load_results
from analysis.tables import latency_table, throughput
from bench.records import drop_warmup
from analysis.goodput import max_slo_concurrency


def decide(sweep: str, gpu_hourly_usd: float, price_source: str, peak_qps: float,
           headroom: float = 0.30, gpus_per_replica: int = 1) -> dict:
    meta, levels = load_results(sweep)
    slo = meta["slo"]
    steady = [{"concurrency": lv["concurrency"],
               "records": drop_warmup(lv["records"], warmup_requests=meta.get("warmup_requests", 0)),
               "duration_s": lv["duration_s"]} for lv in levels]
    c = max_slo_concurrency(steady, slo["ttft_s"], slo["tpot_s"])
    if c is None:
        raise SystemExit("no concurrency level meets the SLO at 95 percent attainment: loosen the SLO or fix the config")
    lv = next(x for x in steady if x["concurrency"] == c)
    thr = throughput(lv["records"])
    lat = latency_table(lv["records"])
    gpus = (meta.get("engine") or {}).get("num_gpus", gpus_per_replica)
    cost = cost_per_million_tokens(thr["output_tok_per_s"], gpu_hourly_usd, num_gpus=gpus)
    mean_e2e = sum(r.e2e_s for r in lv["records"] if r.ok) / max(1, lat["n_ok"])
    plan = fleet_plan(peak_qps, mean_e2e, c, gpus_per_replica=gpus, headroom=headroom)
    return {
        "sweep": meta["name"], "slo": slo, "operating_concurrency": c,
        "output_tok_per_s": thr["output_tok_per_s"], "mean_e2e_s": mean_e2e,
        "gpu_hourly_usd": gpu_hourly_usd, "price_source": price_source, "num_gpus_per_replica": gpus,
        "cost_per_million_output_tokens_usd": cost,
        "peak_qps": peak_qps, "headroom": headroom, "replicas": plan["replicas"], "gpus": plan["gpus"],
        "utilisation_at_peak": plan["utilisation_at_peak"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--gpu-hourly-usd", type=float, required=True)
    ap.add_argument("--price-source", required=True, help="where you read the price, with the date")
    ap.add_argument("--peak-qps", type=float, required=True)
    ap.add_argument("--headroom", type=float, default=0.30)
    ap.add_argument("--out", default="results/decision.json")
    a = ap.parse_args(argv)
    d = decide(a.sweep, a.gpu_hourly_usd, a.price_source, a.peak_qps, a.headroom)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
