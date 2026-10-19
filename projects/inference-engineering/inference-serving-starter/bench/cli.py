"""Command-line glue. GIVEN. It calls the functions YOU wrote; until you write them it fails.

    # closed-loop concurrency sweep against whatever is listening on --base-url
    python -m bench.cli sweep --base-url http://127.0.0.1:8100 --name l4_bf16 \
        --levels 1,2,4,8,16,32 --out results/l4_bf16.json

    # one open-loop run at a target QPS
    python -m bench.cli open --base-url http://127.0.0.1:8100 --name l4_bf16_qps2 \
        --qps 2 --duration 120 --out results/l4_bf16_qps2.json

Works unchanged against a real `vllm serve` (see gpu/). Result file schema:

    {"meta":   {name, base_url, model, time_scale, engine (engine_sim config or null), seed,
                dataset, warmup_requests, slo: {ttft_s, tpot_s}},
     "levels": [{"concurrency": c | "qps": q, "duration_s": sim seconds of the run,
                 "records": [RequestRecord as dict, ...],
                 "server_before": "<prometheus text>", "server_after": "<prometheus text>"}]}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from .client import fetch_info, stream_completion
from .clock import SimClock
from .dataset import DEFAULT_DATASET, load_dataset, sample_workload
from .loadgen import concurrency_sweep, poisson_arrival_times, run_open_loop
from .records import to_record

DEFAULT_SLO = {"ttft_s": 2.0, "tpot_s": 0.15}


async def _text(client, url: str) -> str:
    try:
        r = await client.get(url, timeout=10)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""


async def _model_name(client, base: str) -> str:
    try:
        r = await client.get(base + "/v1/models", timeout=10)
        return r.json()["data"][0]["id"]
    except Exception:
        return "model"


async def _reset_cache(client, base: str) -> None:
    try:
        await client.post(base + "/reset_prefix_cache", timeout=10)
    except Exception:
        pass


async def amain(a) -> int:
    base = a.base_url.rstrip("/")
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    async with httpx.AsyncClient(limits=limits) as client:
        info = await fetch_info(client, base)
        clock = SimClock(info.get("time_scale", 1.0))
        model = a.model or await _model_name(client, base)
        rows = load_dataset(a.dataset)
        levels_out: list = []

        def make_send():
            async def send(item):
                return await stream_completion(client, base, model, item.prompt_ids, item.max_tokens,
                                               item.rid, clock.now())
            return send

        if a.cmd == "sweep":
            levels = [int(x) for x in a.levels.split(",")]
            for c in levels:
                await _reset_cache(client, base)
                before = await _text(client, base + "/metrics")
                t0 = clock.now()
                n = max(a.min_requests, a.requests_per_client * c)
                out = await concurrency_sweep(make_send, lambda _l, n=n: sample_workload(rows, n, a.seed + c), [c])
                t1 = clock.now()
                after = await _text(client, base + "/metrics")
                levels_out.append({"concurrency": c, "duration_s": t1 - t0,
                                   "records": [asdict(to_record(r)) for r in out[c]],
                                   "server_before": before, "server_after": after})
                print(f"  c={c}: {n} requests in {t1 - t0:.1f} sim s", file=sys.stderr, flush=True)
        else:
            await _reset_cache(client, base)
            before = await _text(client, base + "/metrics")
            items = sample_workload(rows, max(1, int(a.qps * a.duration * 1.5)), a.seed)
            times = poisson_arrival_times(a.qps, a.duration, random.Random(a.seed))
            items = items[: len(times)]
            t0 = clock.now()
            shifted = [t0 + t for t in times]
            raws = await run_open_loop(make_send(), items, shifted, clock)
            t1 = clock.now()
            after = await _text(client, base + "/metrics")
            levels_out.append({"qps": a.qps, "duration_s": t1 - t0,
                               "records": [asdict(to_record(r)) for r in raws],
                               "server_before": before, "server_after": after})

    meta = {"name": a.name, "base_url": base, "model": model, "time_scale": clock.time_scale,
            "engine": info.get("config"), "seed": a.seed, "dataset": str(Path(a.dataset).name),
            "warmup_requests": a.warmup_requests, "slo": {"ttft_s": a.slo_ttft, "tpot_s": a.slo_tpot}}
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "levels": levels_out}), encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sweep", "open"):
        s = sub.add_parser(name)
        s.add_argument("--base-url", default="http://127.0.0.1:8100")
        s.add_argument("--model", default=None)
        s.add_argument("--name", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--dataset", default=str(DEFAULT_DATASET))
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--warmup-requests", type=int, default=8)
        s.add_argument("--slo-ttft", type=float, default=DEFAULT_SLO["ttft_s"])
        s.add_argument("--slo-tpot", type=float, default=DEFAULT_SLO["tpot_s"])
        if name == "sweep":
            s.add_argument("--levels", default="1,2,4,8,16,32")
            s.add_argument("--requests-per-client", type=int, default=6)
            s.add_argument("--min-requests", type=int, default=40)
        else:
            s.add_argument("--qps", type=float, required=True)
            s.add_argument("--duration", type=float, default=120.0)
    return asyncio.run(amain(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
