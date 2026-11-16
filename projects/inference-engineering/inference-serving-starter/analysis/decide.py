"""Turn one sweep into a cost and a fleet size and write results/decision.json. GIVEN glue
over the functions YOU wrote.

    python -m analysis.decide --sweep results/l4_fp8.json --gpu-hourly-usd 0.80 \
        --price-source "provider page URL, read 2026-10-20" --peak-qps 40 \
        --api-input-usd-per-m 0.15 --api-output-usd-per-m 0.60 \
        --api-source "API pricing page URL, read 2026-10-20"

--gpu-hourly-usd and the two API prices are YOUR inputs. This repo has no price list, on purpose
(the numbers above are placeholders, not quotes). The operating point is the highest concurrency
that still meets the SLO (analysis.goodput.max_slo_concurrency).

The decision is made under TWO SLOs, because one limit hides the trade. The first (default TTFT
2 s, TPOT 0.15 s) is the sweep's own SLO, an interactive chat limit. The second (default TTFT 10 s,
TPOT 0.3 s, --second-slo-*) is a looser limit for work where nobody watches the first token, such
as batch summarisation. Both are inputs you may change; say in the report why you chose them.

decision.json also holds the input-token and self-host-versus-API figures (see analysis/cost.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cost import (api_cost_per_request, blended_cost_per_million_tokens, breakeven_utilisation,
                           cost_per_million_tokens, cost_per_request)
from analysis.fleet import fleet_plan
from analysis.results_io import load_results
from analysis.tables import latency_table, throughput
from bench.records import drop_warmup
from analysis.goodput import max_slo_concurrency


def _point(steady: list, ttft_s: float, tpot_s: float, gpu_hourly_usd: float, gpus: int, peak_qps: float,
           headroom: float) -> dict:
    c = max_slo_concurrency(steady, ttft_s, tpot_s)
    if c is None:
        raise SystemExit(f"no concurrency level meets TTFT <= {ttft_s} s and TPOT <= {tpot_s} s at 95 percent "
                         "attainment: loosen the SLO or fix the config")
    lv = next(x for x in steady if x["concurrency"] == c)
    thr = throughput(lv["records"])
    lat = latency_table(lv["records"])
    ok = [r for r in lv["records"] if r.ok]
    mean_e2e = sum(r.e2e_s for r in ok) / max(1, lat["n_ok"])
    plan = fleet_plan(peak_qps, mean_e2e, c, gpus_per_replica=gpus, headroom=headroom)
    return {
        "slo": {"ttft_s": ttft_s, "tpot_s": tpot_s}, "operating_concurrency": c,
        "output_tok_per_s": thr["output_tok_per_s"], "req_per_s": thr["req_per_s"], "mean_e2e_s": mean_e2e,
        "mean_prompt_tokens": sum(r.prompt_tokens for r in ok) / max(1, len(ok)),
        "mean_output_tokens": sum(r.output_tokens for r in ok) / max(1, len(ok)),
        "cost_per_million_output_tokens_usd": cost_per_million_tokens(thr["output_tok_per_s"], gpu_hourly_usd,
                                                                      num_gpus=gpus),
        "replicas": plan["replicas"], "gpus": plan["gpus"], "utilisation_at_peak": plan["utilisation_at_peak"],
    }


def decide(sweep: str, gpu_hourly_usd: float, price_source: str, peak_qps: float,
           headroom: float = 0.30, gpus_per_replica: int = 1,
           api_input_usd_per_m: float = 0.0, api_output_usd_per_m: float = 0.0, api_source: str = "",
           second_slo_ttft_s: float = 10.0, second_slo_tpot_s: float = 0.3) -> dict:
    meta, levels = load_results(sweep)
    slo = meta["slo"]
    steady = [{"concurrency": lv["concurrency"],
               "records": drop_warmup(lv["records"], warmup_requests=meta.get("warmup_requests", 0)),
               "duration_s": lv["duration_s"]} for lv in levels]
    gpus = (meta.get("engine") or {}).get("num_gpus", gpus_per_replica)
    first = _point(steady, slo["ttft_s"], slo["tpot_s"], gpu_hourly_usd, gpus, peak_qps, headroom)
    second = _point(steady, second_slo_ttft_s, second_slo_tpot_s, gpu_hourly_usd, gpus, peak_qps, headroom)

    req_cost = cost_per_request(first["req_per_s"], gpu_hourly_usd, num_gpus=gpus)
    api_req = api_cost_per_request(first["mean_prompt_tokens"], first["mean_output_tokens"],
                                   api_input_usd_per_m, api_output_usd_per_m)
    return {
        "sweep": meta["name"], "slo": slo, "operating_concurrency": first["operating_concurrency"],
        "output_tok_per_s": first["output_tok_per_s"], "mean_e2e_s": first["mean_e2e_s"],
        "gpu_hourly_usd": gpu_hourly_usd, "price_source": price_source, "num_gpus_per_replica": gpus,
        "cost_per_million_output_tokens_usd": first["cost_per_million_output_tokens_usd"],
        "peak_qps": peak_qps, "headroom": headroom, "replicas": first["replicas"], "gpus": first["gpus"],
        "utilisation_at_peak": first["utilisation_at_peak"],
        # Second SLO row: the same sweep read under a looser limit.
        "second_slo": {k: second[k] for k in ("slo", "operating_concurrency", "output_tok_per_s",
                                              "cost_per_million_output_tokens_usd", "replicas", "gpus")},
        # Input-token cost and the self-host-versus-API line, at the first SLO's operating point.
        "mean_prompt_tokens": first["mean_prompt_tokens"], "mean_output_tokens": first["mean_output_tokens"],
        "cost_per_request_usd": req_cost,
        "cost_per_million_all_tokens_usd": blended_cost_per_million_tokens(
            req_cost, first["mean_prompt_tokens"], first["mean_output_tokens"]),
        "api": {"input_usd_per_m": api_input_usd_per_m, "output_usd_per_m": api_output_usd_per_m,
                "source": api_source, "cost_per_request_usd": api_req,
                "breakeven_utilisation": breakeven_utilisation(req_cost, api_req) if api_req > 0 else None},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--gpu-hourly-usd", type=float, required=True)
    ap.add_argument("--price-source", required=True, help="where you read the price, with the date")
    ap.add_argument("--peak-qps", type=float, required=True)
    ap.add_argument("--headroom", type=float, default=0.30)
    ap.add_argument("--api-input-usd-per-m", type=float, required=True, help="hosted API price per 1M input tokens")
    ap.add_argument("--api-output-usd-per-m", type=float, required=True, help="hosted API price per 1M output tokens")
    ap.add_argument("--api-source", required=True, help="where you read the API prices, with the date")
    ap.add_argument("--second-slo-ttft", type=float, default=10.0)
    ap.add_argument("--second-slo-tpot", type=float, default=0.3)
    ap.add_argument("--out", default="results/decision.json")
    a = ap.parse_args(argv)
    d = decide(a.sweep, a.gpu_hourly_usd, a.price_source, a.peak_qps, a.headroom,
               api_input_usd_per_m=a.api_input_usd_per_m, api_output_usd_per_m=a.api_output_usd_per_m,
               api_source=a.api_source, second_slo_ttft_s=a.second_slo_ttft, second_slo_tpot_s=a.second_slo_tpot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
