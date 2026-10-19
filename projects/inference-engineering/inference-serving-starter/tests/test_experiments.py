"""GATE 4: you ran the experiments and the results are physically sensible.

Needs results/quant_bf16.json, quant_fp8.json, quant_int4.json, spec_off.json,
spec_ngram.json (python scripts/run_experiments.py) and at least one results/knee_*.png
(python -m analysis.summarize results/quant_bf16.json --plot results/knee_quant_bf16.png).
Every assertion below is something the physics says must be true. A harness bug makes one of
them fail, so a red line here is information about YOUR code.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from analysis.goodput import goodput
from analysis.knee import find_knee
from analysis.results_io import load_results
from analysis.tables import latency_table, throughput
from bench.records import drop_warmup

RESULTS = Path(__file__).resolve().parent.parent / "results"
NEED = ["quant_bf16", "quant_fp8", "quant_int4", "spec_off", "spec_ngram"]


def _tput(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        pytest.fail(f"results/{name}.json is missing: run `python scripts/run_experiments.py` once bench/ works")
    meta, levels = load_results(p)
    out = {}
    for lv in levels:
        recs = drop_warmup(lv["records"], warmup_requests=meta["warmup_requests"])
        out[lv["concurrency"]] = {"tok_s": throughput(recs)["output_tok_per_s"], "recs": recs,
                                  "lat": latency_table(recs), "meta": meta}
    return out


@pytest.fixture(scope="module")
def data():
    return {n: _tput(n) for n in NEED}


def test_every_experiment_has_the_expected_levels_and_no_failures(data):
    for name, lv in data.items():
        assert len(lv) >= 6, name
        for c, d in lv.items():
            assert d["lat"]["n_failed"] == 0, f"{name} c={c} has failed requests"
            assert d["lat"]["n_ok"] >= 20, f"{name} c={c}: too few requests left after warm-up"
            assert d["lat"]["ttft_s"][50] > 0 and d["lat"]["e2e_s"][50] > d["lat"]["ttft_s"][50]


def test_throughput_rises_with_concurrency_then_flattens(data):
    t = data["quant_bf16"]
    assert t[4]["tok_s"] > 2.5 * t[1]["tok_s"]
    assert t[16]["tok_s"] > 1.5 * t[4]["tok_s"]
    levels = sorted(t)
    knee = find_knee([(c, t[c]["tok_s"]) for c in levels])
    assert levels[0] < knee < levels[-1], "a real curve has a knee strictly inside the range"


def test_latency_blows_up_past_the_knee(data):
    t = data["quant_bf16"]
    top = max(t)
    assert t[top]["lat"]["ttft_s"][99] > 3 * t[1]["lat"]["ttft_s"][99]
    assert t[top]["lat"]["itl_s"][99] >= t[1]["lat"]["itl_s"][99]


def test_quantisation_speeds_up_small_batch_decode(data):
    b, f, i = (data[n][1]["tok_s"] for n in ("quant_bf16", "quant_fp8", "quant_int4"))
    assert f > 1.4 * b and i > 1.8 * b, f"bf16 {b:.1f}, fp8 {f:.1f}, int4 {i:.1f} tokens/s at concurrency 1"


def test_quantisation_also_buys_kv_room_at_high_concurrency(data):
    top = max(data["quant_bf16"])
    b, f = data["quant_bf16"][top]["tok_s"], data["quant_fp8"][top]["tok_s"]
    assert f > 1.3 * b, "smaller weights leave more memory for KV cache, so more sequences run at once"


def test_slo_goodput_is_below_throughput_where_the_server_is_overloaded(data):
    top = max(data["quant_bf16"])
    meta = data["quant_bf16"][top]["meta"]
    recs = data["quant_bf16"][top]["recs"]
    gp = goodput(recs, meta["slo"]["ttft_s"], meta["slo"]["tpot_s"])
    assert gp < 0.8 * throughput(recs)["req_per_s"]
    light = data["quant_bf16"][1]["recs"]
    assert goodput(light, meta["slo"]["ttft_s"], meta["slo"]["tpot_s"]) > 0.9 * throughput(light)["req_per_s"]


def test_speculation_helps_when_memory_bound_and_hurts_when_compute_bound(data):
    off, on = data["spec_off"], data["spec_ngram"]
    lo = min(off)
    hi = max(off)
    assert on[lo]["tok_s"] > 1.3 * off[lo]["tok_s"], f"at concurrency {lo} speculation should clearly win"
    assert on[hi]["tok_s"] < off[hi]["tok_s"], f"at concurrency {hi} speculation should lose"


def test_there_is_a_knee_plot():
    pngs = list(RESULTS.glob("knee_*.png"))
    if not pngs:
        pytest.fail("no results/knee_*.png: python -m analysis.summarize results/quant_bf16.json --plot results/knee_quant_bf16.png")
    assert pngs[0].read_bytes()[:4] == b"\x89PNG"
