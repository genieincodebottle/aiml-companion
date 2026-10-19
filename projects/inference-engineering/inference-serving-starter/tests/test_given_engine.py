"""GATE 1: the given parts behave. This file PASSES on a fresh clone. If it does not, your
environment is broken, not your code. Do not edit it.

What it pins down is the physics you will rely on in the report: decode throughput rises with
batch until the KV cache is full, prefix hits cut TTFT, quantisation speeds up small-batch
decode, speculation wins when memory-bound and loses when compute-bound.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from engine_sim.config import EngineConfig
from engine_sim.core import Engine, Request
from helpers import closed_loop

ROOT = Path(__file__).resolve().parent.parent


def test_throughput_rises_with_batch_until_kv_is_full():
    cfg = EngineConfig(gpu="l4", quant="bf16")
    tput = {c: closed_loop(cfg, c, n=max(30, 3 * c))["tok_per_s"] for c in (1, 4, 16)}
    assert tput[4] > 2.5 * tput[1], "batching should amortise the weight read"
    assert tput[16] > 2.0 * tput[4]
    heavy = closed_loop(cfg, 96, n=150)
    assert heavy["preemptions"] > 0, "96 concurrent 750-token contexts do not fit in an L4's KV cache"
    assert closed_loop(cfg, 4, n=30)["preemptions"] == 0


def test_prefix_cache_hit_cuts_ttft_and_counts_hits():
    cfg = EngineConfig(gpu="l4")
    eng = Engine(cfg)
    ids = list(range(1000, 3000))
    eng.submit(Request("a", 0.0, ids, 8))
    ev = eng.run_until(1e6)
    first = next(e for e in ev if e.rid == "a" and e.first)
    ttft_cold = first.t - first.arrival
    eng.submit(Request("b", eng.now + 1.0, ids + [5, 6, 7], 8))
    ev = eng.run_until(1e6)
    second = next(e for e in ev if e.rid == "b" and e.first)
    ttft_warm = second.t - second.arrival
    assert second.cached_tokens >= 1900
    assert ttft_warm < 0.5 * ttft_cold
    m = eng.metrics
    assert m.prefix_hits >= 1900 and m.prefix_queries >= 4000


def test_int4_decodes_faster_than_bf16_at_batch_one():
    bf16 = closed_loop(EngineConfig(gpu="l4", quant="bf16"), 1, n=20)["tok_per_s"]
    fp8 = closed_loop(EngineConfig(gpu="l4", quant="fp8"), 1, n=20)["tok_per_s"]
    int4 = closed_loop(EngineConfig(gpu="l4", quant="int4"), 1, n=20)["tok_per_s"]
    assert fp8 > 1.5 * bf16
    assert int4 > 2.0 * bf16


def test_speculation_wins_when_memory_bound_and_loses_when_compute_bound():
    base = EngineConfig(gpu="a10", quant="bf16", spec_accept=0.5, max_num_seqs=128)
    off, on = base, base.with_(spec="ngram")
    low_off = closed_loop(off, 2, n=30, prompt_len=100, out_len=120)["tok_per_s"]
    low_on = closed_loop(on, 2, n=30, prompt_len=100, out_len=120)["tok_per_s"]
    hi_off = closed_loop(off, 128, n=256, prompt_len=100, out_len=120)["tok_per_s"]
    hi_on = closed_loop(on, 128, n=256, prompt_len=100, out_len=120)["tok_per_s"]
    assert low_on > 1.3 * low_off
    assert hi_on < hi_off


def test_unsupported_configs_fail_loudly():
    with pytest.raises(ValueError, match="no FP8"):
        EngineConfig(gpu="a10", quant="fp8").validate()
    with pytest.raises(ValueError, match="almost nothing is left for KV cache"):
        EngineConfig(model="70b", gpu="l4", quant="bf16").validate()
    EngineConfig(model="70b", gpu="h100", num_gpus=4, quant="bf16").validate()


def test_same_seed_same_bytes():
    cfg = EngineConfig(gpu="a10", spec="ngram", seed=5)
    a = closed_loop(cfg, 8, n=40)["engine"].render_metrics()
    b = closed_loop(cfg, 8, n=40)["engine"].render_metrics()
    assert a == b


def test_metrics_use_vllm_names():
    text = closed_loop(EngineConfig(gpu="l4"), 4, n=12)["engine"].render_metrics()
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
                 "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                 "vllm:time_to_first_token_seconds_bucket", "vllm:inter_token_latency_seconds_count",
                 "vllm:e2e_request_latency_seconds_sum", "vllm:spec_decode_num_accepted_tokens_total"):
        assert name in text, name


def test_http_streaming_roundtrip(start_engine):
    base = start_engine("--gpu", "l4", time_scale=100.0)
    body = {"prompt": list(range(200)), "max_tokens": 12, "stream": True,
            "stream_options": {"include_usage": True}}
    chunks, saw_done = [], False
    with httpx.stream("POST", base + "/v1/completions", json=body, timeout=30) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line == "data: [DONE]":
                saw_done = True
            elif line.startswith("data: "):
                chunks.append(json.loads(line[6:]))
    content = [c for c in chunks if c.get("choices") and c["choices"][0].get("text")]
    steps = [s for c in content for s in c["x_steps"]]
    assert saw_done and sum(n for _, n in steps) == 12 and sum(c["x_n_tokens"] for c in content) == 12
    sim_t = [t for t, _ in steps]
    assert sim_t == sorted(sim_t) and sim_t[0] > content[0]["x_sim_arrival"]
    assert chunks[-1]["usage"]["completion_tokens"] == 12
    text = httpx.get(base + "/metrics").text
    assert "vllm:num_requests_running" in text
    assert httpx.get(base + "/sim/info").json()["engine_sim"] is True
    chat = httpx.post(base + "/v1/chat/completions", timeout=30,
                      json={"messages": [{"role": "user", "content": "hello there"}], "max_tokens": 5}).json()
    assert chat["usage"]["completion_tokens"] == 5


def test_incident_bundles_regenerate_byte_for_byte(tmp_path):
    out = tmp_path / "inc"
    subprocess.run([sys.executable, str(ROOT / "debug_lab" / "make_incidents.py"), "--out", str(out)],
                   check=True, capture_output=True, cwd=ROOT)
    committed = ROOT / "debug_lab" / "incidents"
    files = sorted(p.relative_to(committed) for p in committed.rglob("*") if p.is_file())
    assert len(files) > 60
    for rel in files:
        assert (out / rel).read_bytes() == (committed / rel).read_bytes(), f"{rel} differs"


def test_gpu_scripts_pin_a_version_and_use_current_flags():
    gpu = ROOT / "gpu"
    text = "\n".join(p.read_text(encoding="utf-8") for p in gpu.rglob("*") if p.is_file())
    assert re.search(r"vllm==0\.31\.0|VLLM_VERSION=0\.31\.0", text), "pin a vLLM version"
    assert "--speculative-config" in text
    for removed in ("--speculative-model", "--num-speculative-tokens", "--use-v2-block-manager",
                    "--ngram-prompt-lookup-max"):
        assert removed not in text, f"{removed} is a removed flag"
    assert "vllm bench serve" in text


def test_k8s_manifests_parse_and_hpa_metric_matches_the_adapter_rule():
    yaml = pytest.importorskip("yaml")
    k8s = ROOT / "k8s"
    docs = []
    for name in ("engine-sim.yaml", "hpa.yaml", "prometheus-adapter-values.yaml", "kind-config.yaml"):
        docs += [d for d in yaml.safe_load_all((k8s / name).read_text(encoding="utf-8")) if d]
    kinds = {d.get("kind") for d in docs}
    assert {"Deployment", "Service", "ServiceMonitor", "HorizontalPodAutoscaler", "Cluster"} <= kinds
    hpa = next(d for d in docs if d.get("kind") == "HorizontalPodAutoscaler")
    metric = hpa["spec"]["metrics"][0]["pods"]["metric"]["name"]
    values = next(d for d in docs if "rules" in d)
    assert values["rules"]["custom"][0]["name"]["as"] == metric
