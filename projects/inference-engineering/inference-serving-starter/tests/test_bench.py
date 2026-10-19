"""GATE 2: the load generator and the per-request records. Fails until bench/ is written."""
from __future__ import annotations

import asyncio
import random
import statistics
import sys
from pathlib import Path

import httpx
import pytest

from bench.client import RawResult
from bench.clock import SimClock
from bench.dataset import load_dataset, sample_workload
from bench.loadgen import (concurrency_sweep, poisson_arrival_times, run_closed_loop, run_open_loop)
from bench.records import RequestRecord, drop_warmup, to_record

ROOT = Path(__file__).resolve().parent.parent


# ---- dataset -------------------------------------------------------------------------
def test_sample_workload_is_deterministic_and_matches_the_dataset():
    rows = load_dataset()
    a = sample_workload(rows, 600, seed=3)
    b = sample_workload(rows, 600, seed=3)
    c = sample_workload(rows, 600, seed=4)
    assert [x.prompt_ids for x in a] == [x.prompt_ids for x in b]
    assert [x.max_tokens for x in a] != [x.max_tokens for x in c]
    assert [x.rid for x in a] == list(range(600))
    med_ds = statistics.median(r["prompt_tokens"] for r in rows)
    med_s = statistics.median(len(x.prompt_ids) for x in a)
    assert abs(med_s - med_ds) / med_ds < 0.15
    share_ds = sum(r["prefix_group"] is not None for r in rows) / len(rows)
    share_s = sum(x.prefix_group is not None for x in a) / len(a)
    assert abs(share_s - share_ds) < 0.07


def test_prefix_groups_share_real_prefixes():
    items = sample_workload(load_dataset(), 400, seed=1)
    g0 = [x for x in items if x.prefix_group == 0]
    assert len(g0) > 5
    assert g0[0].prompt_ids[:192] == g0[1].prompt_ids[:192]
    assert g0[0].prompt_ids[200:260] != g0[1].prompt_ids[200:260]


# ---- arrivals ------------------------------------------------------------------------
def test_poisson_arrivals_are_exponential_not_uniform():
    t = poisson_arrival_times(5.0, 4000.0, random.Random(1))
    assert t == sorted(t) and t[-1] < 4000.0
    gaps = [b - a for a, b in zip(t, t[1:])]
    mean, sd = statistics.mean(gaps), statistics.pstdev(gaps)
    assert abs(mean - 0.2) < 0.01
    assert 0.9 < sd / mean < 1.1, "an exponential gap has std equal to its mean; fixed spacing has std 0"
    assert poisson_arrival_times(5.0, 100.0, random.Random(9)) == poisson_arrival_times(5.0, 100.0, random.Random(9))


# ---- closed and open loop ------------------------------------------------------------
def _fake_send(counter: dict, delay_wall: float, clock: SimClock | None = None, log: list | None = None):
    async def send(item):
        counter["in"] += 1
        counter["max"] = max(counter["max"], counter["in"])
        if log is not None:
            log.append((item.rid, clock.now()))
        await asyncio.sleep(delay_wall)
        counter["in"] -= 1
        return RawResult(rid=item.rid, prompt_tokens=len(item.prompt_ids), max_tokens=item.max_tokens,
                         send_sim=0.0, ok=True)
    return send


def test_closed_loop_holds_exactly_the_concurrency():
    items = sample_workload(load_dataset(), 40, seed=1)
    counter = {"in": 0, "max": 0}
    out = asyncio.run(run_closed_loop(_fake_send(counter, 0.01), items, 4))
    assert counter["max"] == 4
    assert [r.rid for r in out] == [it.rid for it in items]
    with pytest.raises(ValueError):
        asyncio.run(run_closed_loop(_fake_send(counter, 0.0), items, 0))


def test_open_loop_keeps_the_schedule_while_requests_are_slow():
    items = sample_workload(load_dataset(), 20, seed=1)
    clock = SimClock(20.0)
    t0 = clock.now()
    times = [t0 + 0.5 * i for i in range(20)]       # 2 per sim second for 10 sim seconds
    counter = {"in": 0, "max": 0}
    log: list = []
    out = asyncio.run(run_open_loop(_fake_send(counter, 0.1, clock, log), items, times, clock))  # 0.1 wall = 2 sim s
    assert len(out) == 20 and [r.rid for r in out] == list(range(20))
    assert counter["max"] >= 3, "an open loop must not wait for earlier requests"
    late = [abs(t - times[rid]) for rid, t in log]
    assert max(late) < 1.0, f"requests drifted off schedule by {max(late):.2f} sim s"
    with pytest.raises(ValueError):
        asyncio.run(run_open_loop(_fake_send(counter, 0.0), items, times[:5], clock))


def test_concurrency_sweep_runs_levels_in_order_with_fresh_senders():
    made, seen = [], []

    def make_send():
        made.append(1)
        counter = {"in": 0, "max": 0}

        async def send(item):
            counter["in"] += 1
            seen.append((len(made), counter["in"]))
            await asyncio.sleep(0.005)
            counter["in"] -= 1
            return RawResult(rid=item.rid, prompt_tokens=1, max_tokens=1, send_sim=0.0, ok=True)
        return send

    rows = load_dataset()
    out = asyncio.run(concurrency_sweep(make_send, lambda c: sample_workload(rows, 6 * c, seed=c), [1, 2, 8]))
    assert list(out) == [1, 2, 8] and [len(out[c]) for c in (1, 2, 8)] == [6, 12, 48]
    assert len(made) == 3
    assert max(n for lvl, n in seen if lvl == 1) == 1 and max(n for lvl, n in seen if lvl == 3) == 8


# ---- records -------------------------------------------------------------------------
def test_to_record_uses_engine_time_when_present():
    raw = RawResult(rid=7, prompt_tokens=100, max_tokens=5, send_sim=3.0, wall_start=10.0, wall_end=10.5, ok=True,
                    chunk_wall=[10.1, 10.2, 10.3, 10.4, 10.5], chunk_sim=[5.0, 5.2, 5.4, 5.6, 6.0],
                    chunk_tokens=[1, 1, 1, 1, 1], sim_arrival=4.0, completion_tokens=5)
    rec = to_record(raw)
    assert isinstance(rec, RequestRecord) and rec.ok and rec.rid == 7 and rec.start_s == 3.0
    assert rec.ttft_s == pytest.approx(1.0) and rec.e2e_s == pytest.approx(2.0)
    assert rec.itl_s == pytest.approx([0.2, 0.2, 0.2, 0.4])
    assert rec.tpot_s == pytest.approx((2.0 - 1.0) / 4)
    assert rec.output_tokens == 5 and rec.prompt_tokens == 100


def test_to_record_falls_back_to_wall_time_and_prefers_usage_for_tokens():
    raw = RawResult(rid=1, prompt_tokens=10, max_tokens=8, send_sim=0.0, wall_start=100.0, wall_end=101.0, ok=True,
                    chunk_wall=[100.4, 100.6, 101.0], completion_tokens=8)   # speculation: 3 chunks, 8 tokens
    rec = to_record(raw)
    assert rec.ttft_s == pytest.approx(0.4) and rec.e2e_s == pytest.approx(1.0)
    assert rec.output_tokens == 8, "chunk count undercounts under speculation; use the usage block"
    assert rec.tpot_s == pytest.approx(0.6 / 7)
    one = to_record(RawResult(rid=2, prompt_tokens=1, max_tokens=1, send_sim=0.0, wall_start=0.0, wall_end=0.3,
                              ok=True, chunk_wall=[0.3], completion_tokens=1))
    assert one.tpot_s == 0.0 and one.itl_s == []


def test_failed_request_becomes_a_failed_record():
    rec = to_record(RawResult(rid=3, prompt_tokens=5, max_tokens=5, send_sim=1.0, ok=False, error="HTTP 500"))
    assert rec.ok is False and rec.ttft_s == 0.0 and rec.rid == 3


def test_drop_warmup_trims_start_and_end_without_mutating():
    recs = [RequestRecord(rid=i, start_s=float(i), ttft_s=1, e2e_s=2, tpot_s=0.1) for i in (5, 0, 1, 2, 3, 4, 6, 7, 8, 9)]
    snapshot = list(recs)
    kept = drop_warmup(recs, warmup_requests=2, cooldown_requests=1)
    assert [r.rid for r in kept] == [2, 3, 4, 5, 6, 7, 8]
    assert recs == snapshot
    assert [r.rid for r in drop_warmup(recs, warmup_s=3.0)] == [3, 4, 5, 6, 7, 8, 9]
    assert drop_warmup(recs, warmup_requests=50) == []


# ---- the harness against a live engine: does it match the server's own books? ---------
def test_harness_agrees_with_the_servers_histograms(start_engine, tmp_path):
    import json
    import subprocess

    base = start_engine("--gpu", "l4", "--quant", "bf16", time_scale=50.0)
    out = tmp_path / "live.json"
    r = subprocess.run([sys.executable, "-m", "bench.cli", "sweep", "--base-url", base, "--name", "live",
                        "--levels", "2,4", "--requests-per-client", "5", "--min-requests", "10",
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    sys.path.insert(0, str(ROOT / "debug_lab"))
    from promparse import hist_mean, metric_value
    data = json.loads(out.read_text())
    assert [lv["concurrency"] for lv in data["levels"]] == [2, 4]
    for lv in data["levels"]:
        recs = lv["records"]
        assert recs and all(x["ok"] for x in recs)
        n = len(recs)
        b, a = lv["server_before"], lv["server_after"]
        d_cnt = metric_value(a, "vllm:time_to_first_token_seconds_count") - (metric_value(b, "vllm:time_to_first_token_seconds_count") or 0)
        assert d_cnt == n
        d_sum = metric_value(a, "vllm:time_to_first_token_seconds_sum") - (metric_value(b, "vllm:time_to_first_token_seconds_sum") or 0)
        mine = sum(x["ttft_s"] for x in recs) / n
        assert abs(mine - d_sum / n) / (d_sum / n) < 0.05, "your TTFT disagrees with the server's own histogram"
        d_tok = metric_value(a, "vllm:generation_tokens_total") - (metric_value(b, "vllm:generation_tokens_total") or 0)
        assert d_tok == sum(x["output_tokens"] for x in recs)
