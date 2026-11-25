"""Turn one sweep into a GPU-hours figure and a fleet size and write results/decision.json. GIVEN
glue over the functions YOU wrote.

    python -m analysis.decide --sweep results/l4_fp8.json --peak-qps 40 \
        --api-input-gpu-h-per-m 1.0 --api-output-gpu-h-per-m 3.0 \
        --api-basis "how you turned the API's rate into GPU-hours of your GPU, and when"

The repo works in GPU time and holds no price. The two --api-*-gpu-h-per-m values are YOUR inputs:
the API's rate for 1M input (or output) tokens divided by the hourly rate of the GPU you ran, so each
reads "GPU-hours of my GPU that 1M API tokens are worth" (the numbers above are placeholders, not
quotes). The self-host-versus-API question is answered as GPU-seconds per request, plus the
utilisation at which the two tie (analysis.cost.breakeven_utilisation). The operating point is the
highest concurrency that still meets the SLO (analysis.goodput.max_slo_concurrency).

The decision is made under TWO SLOs, because one limit hides the trade. The first (default TTFT
2 s, TPOT 0.15 s) is the sweep's own SLO, an interactive chat limit. The second (default TTFT 10 s,
TPOT 0.3 s, --second-slo-*) is a looser limit for work where nobody watches the first token, such
as batch summarisation. Both are inputs you may change; say in the report why you chose them.

decision.json also holds the effective (utilisation-adjusted) GPU-hours, the input-token and the
self-host-versus-API figures (see analysis/cost.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.cost import (api_gpu_seconds_per_request, blended_gpu_hours_per_million_tokens,
                           breakeven_utilisation, gpu_hours_per_million_tokens, gpu_seconds_per_request)
from analysis.fleet import fleet_plan
from analysis.results_io import load_results
from analysis.tables import latency_table, throughput
from bench.records import drop_warmup
from analysis.goodput import max_slo_concurrency


def _point(steady: list, ttft_s: float, tpot_s: float, gpus: int, peak_qps: float, headroom: float) -> dict:
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
        "gpu_hours_per_million_output_tokens": gpu_hours_per_million_tokens(thr["output_tok_per_s"], num_gpus=gpus),
        "replicas": plan["replicas"], "gpus": plan["gpus"], "utilisation_at_peak": plan["utilisation_at_peak"],
    }


def decide(sweep: str, peak_qps: float, headroom: float = 0.30, gpus_per_replica: int = 1,
           api_input_gpu_h_per_m: float = 0.0, api_output_gpu_h_per_m: float = 0.0, api_basis: str = "",
           second_slo_ttft_s: float = 10.0, second_slo_tpot_s: float = 0.3) -> dict:
    meta, levels = load_results(sweep)
    slo = meta["slo"]
    steady = [{"concurrency": lv["concurrency"],
               "records": drop_warmup(lv["records"], warmup_requests=meta.get("warmup_requests", 0)),
               "duration_s": lv["duration_s"]} for lv in levels]
    gpus = (meta.get("engine") or {}).get("num_gpus", gpus_per_replica)
    first = _point(steady, slo["ttft_s"], slo["tpot_s"], gpus, peak_qps, headroom)
    second = _point(steady, second_slo_ttft_s, second_slo_tpot_s, gpus, peak_qps, headroom)

    req_gpu_s = gpu_seconds_per_request(first["req_per_s"], num_gpus=gpus)
    api_req = api_gpu_seconds_per_request(first["mean_prompt_tokens"], first["mean_output_tokens"],
                                          api_input_gpu_h_per_m, api_output_gpu_h_per_m)
    # Effective GPU-hours: the same throughput, charged for the idle share of the fleet at peak.
    u = first["utilisation_at_peak"]
    effective = gpu_hours_per_million_tokens(first["output_tok_per_s"], num_gpus=gpus, utilisation=u) \
        if 0 < u <= 1 else None
    return {
        "sweep": meta["name"], "slo": slo, "operating_concurrency": first["operating_concurrency"],
        "output_tok_per_s": first["output_tok_per_s"], "mean_e2e_s": first["mean_e2e_s"],
        "num_gpus_per_replica": gpus,
        "gpu_hours_per_million_output_tokens": first["gpu_hours_per_million_output_tokens"],
        "effective_gpu_hours_per_million_output_tokens": effective,
        "peak_qps": peak_qps, "headroom": headroom, "replicas": first["replicas"], "gpus": first["gpus"],
        "utilisation_at_peak": first["utilisation_at_peak"],
        # Second SLO row: the same sweep read under a looser limit.
        "second_slo": {k: second[k] for k in ("slo", "operating_concurrency", "output_tok_per_s",
                                              "gpu_hours_per_million_output_tokens", "replicas", "gpus")},
        # Input-token usage and the self-host-versus-API line, at the first SLO's operating point.
        "mean_prompt_tokens": first["mean_prompt_tokens"], "mean_output_tokens": first["mean_output_tokens"],
        "gpu_seconds_per_request": req_gpu_s,
        "gpu_hours_per_million_all_tokens": blended_gpu_hours_per_million_tokens(
            req_gpu_s, first["mean_prompt_tokens"], first["mean_output_tokens"]),
        "api": {"input_gpu_h_per_m": api_input_gpu_h_per_m, "output_gpu_h_per_m": api_output_gpu_h_per_m,
                "basis": api_basis, "gpu_seconds_per_request": api_req,
                "breakeven_utilisation": breakeven_utilisation(req_gpu_s, api_req) if api_req > 0 else None},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--peak-qps", type=float, required=True)
    ap.add_argument("--headroom", type=float, default=0.30)
    ap.add_argument("--api-input-gpu-h-per-m", type=float, required=True,
                    help="GPU-hours of your GPU that 1M hosted-API input tokens are worth")
    ap.add_argument("--api-output-gpu-h-per-m", type=float, required=True,
                    help="GPU-hours of your GPU that 1M hosted-API output tokens are worth")
    ap.add_argument("--api-basis", required=True, help="how you derived those two ratios, with the date")
    ap.add_argument("--second-slo-ttft", type=float, default=10.0)
    ap.add_argument("--second-slo-tpot", type=float, default=0.3)
    ap.add_argument("--out", default="results/decision.json")
    a = ap.parse_args(argv)
    d = decide(a.sweep, a.peak_qps, a.headroom,
               api_input_gpu_h_per_m=a.api_input_gpu_h_per_m, api_output_gpu_h_per_m=a.api_output_gpu_h_per_m,
               api_basis=a.api_basis, second_slo_ttft_s=a.second_slo_ttft, second_slo_tpot_s=a.second_slo_tpot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(json.dumps(d, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
