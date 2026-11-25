"""GATE 1 (second file): the given extras behave. PASSES on a fresh clone. Do not edit it.

Covers the parts that are not engine physics: the diagnosis checker cannot be gamed with one
scrape and "FIX: x", the GPU-time helpers do their arithmetic, the optional quality comparator and
its recording script work, the TP=2 script has its flag, the gang-lab manifest and its scheduler
model agree, and the optional profiler script runs. Nothing here needs a GPU or a cluster.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "debug_lab"))
sys.path.insert(0, str(ROOT / "quality"))
sys.path.insert(0, str(ROOT / "k8s" / "gang_lab"))
import check_diagnosis as cd  # noqa: E402
import check_equivalence as ce  # noqa: E402
import run_prompts  # noqa: E402
from promparse import metric_value  # noqa: E402

INC = ROOT / "debug_lab" / "incidents"


# ---- the Gate 5 checker -------------------------------------------------------------------
def _v(incident: str, rel: str, name: str) -> str:
    text = (INC / incident / rel).read_text(encoding="utf-8")
    return f"{metric_value(text, name):.8g}"


def _diag(tmp_path: Path, incident: str, cause: str, evidence: list, fix: str) -> cd.Result:
    body = f"ROOT_CAUSE: {cause}\nEVIDENCE:\n" + "".join(f"- {e}\n" for e in evidence) + f"FIX: {fix}\n"
    (tmp_path / f"{incident}.md").write_text(body, encoding="utf-8")
    return cd.check(incident, diag_dir=tmp_path)


A = "a_kv_saturation"
A_FIX = ("raise the KV budget or cap admission. Confirm: vllm:num_preemptions_total stops rising and "
         "vllm:kv_cache_usage_perc falls below 0.9.")


def _a_evidence(first="scrapes/scrape_08_t0270s.prom", second="scrapes/scrape_10_t0330s.prom") -> list:
    return [f"vllm:kv_cache_usage_perc | {first} | {_v(A, first, 'vllm:kv_cache_usage_perc')}",
            f"vllm:num_preemptions_total | {first} | {_v(A, first, 'vllm:num_preemptions_total')}",
            f"vllm:num_preemptions_total | {second} | {_v(A, second, 'vllm:num_preemptions_total')}"]


def test_a_complete_diagnosis_passes(tmp_path):
    res = _diag(tmp_path, A, "the KV cache is saturated, so the engine preempts", _a_evidence(), A_FIX)
    assert res.ok, res.reasons


def test_one_scrape_and_a_bare_fix_is_rejected(tmp_path):
    """The audit's probe: one scrape and 'FIX: x' used to pass."""
    first = "scrapes/scrape_10_t0330s.prom"
    ev = [f"vllm:kv_cache_usage_perc | {first} | {_v(A, first, 'vllm:kv_cache_usage_perc')}",
          f"vllm:num_preemptions_total | {first} | {_v(A, first, 'vllm:num_preemptions_total')}"]
    res = _diag(tmp_path, A, "kv cache full, preemptions", ev, "x")
    text = " ".join(res.reasons)
    assert not res.ok
    assert "different scrape files" in text and "counter" in text and "confirm" in text


def test_a_counter_read_from_one_scrape_is_rejected(tmp_path):
    f1, f2 = "scrapes/scrape_08_t0270s.prom", "scrapes/scrape_10_t0330s.prom"
    ev = [f"vllm:kv_cache_usage_perc | {f1} | {_v(A, f1, 'vllm:kv_cache_usage_perc')}",
          f"vllm:kv_cache_usage_perc | {f2} | {_v(A, f2, 'vllm:kv_cache_usage_perc')}",
          f"vllm:num_preemptions_total | {f2} | {_v(A, f2, 'vllm:num_preemptions_total')}"]
    res = _diag(tmp_path, A, "the KV cache is saturated, so the engine preempts", ev, A_FIX)
    assert not res.ok and any("is a counter" in r for r in res.reasons)


@pytest.mark.parametrize("fix, expect", [
    ("add GPU memory", "confirm"),                                                   # no confirmation at all
    ("add GPU memory and confirm it worked", "name the metric"),                    # no metric
    ("add GPU memory and confirm vllm:num_preemptions_total", "which way"),         # no direction
])
def test_the_fix_must_say_how_to_confirm_it(tmp_path, fix, expect):
    res = _diag(tmp_path, A, "the KV cache is saturated, so the engine preempts", _a_evidence(), fix)
    assert not res.ok and any(expect in r for r in res.reasons), res.reasons


def test_the_fix_block_may_continue_on_the_next_line(tmp_path):
    fix = "raise the KV budget.\nConfirm that vllm:num_preemptions_total stops rising afterwards."
    res = _diag(tmp_path, A, "the KV cache is saturated, so the engine preempts", _a_evidence(), fix)
    assert res.ok, res.reasons


D = "d_config_regression"


def test_the_optional_regression_incident_is_checked_when_written(tmp_path):
    assert D in cd.SPEC and D in cd.OPTIONAL
    b, a = "before/scrapes/scrape_10_t0330s.prom", "after/scrapes/scrape_10_t0330s.prom"
    a2 = "after/scrapes/scrape_12_t0390s.prom"
    ev = [f"vllm:num_requests_running | {a} | {_v(D, a, 'vllm:num_requests_running')}",
          f"vllm:num_requests_waiting | {a} | {_v(D, a, 'vllm:num_requests_waiting')}",
          f"vllm:kv_cache_usage_perc | {a} | {_v(D, a, 'vllm:kv_cache_usage_perc')}",
          f"vllm:num_requests_waiting | {b} | {_v(D, b, 'vllm:num_requests_waiting')}",
          f"vllm:generation_tokens_total | {a} | {_v(D, a, 'vllm:generation_tokens_total')}",
          f"vllm:generation_tokens_total | {a2} | {_v(D, a2, 'vllm:generation_tokens_total')}"]
    fix = ("restore --max-num-seqs to 128. Confirm vllm:num_requests_waiting falls to near zero and "
           "vllm:num_requests_running rises above 16 under the same traffic.")
    res = _diag(tmp_path, D, "the max-num-seqs limit of 16 caps the running batch, so a queue builds", ev, fix)
    assert res.ok, res.reasons
    only_after = [e for e in ev if "before/" not in e]
    res = _diag(tmp_path, D, "the max-num-seqs limit of 16 caps the running batch, so a queue builds", only_after, fix)
    assert not res.ok and any("both folders" in r for r in res.reasons)


# ---- the GPU-time helpers ---------------------------------------------------------------------
def test_gpu_time_helpers_do_their_arithmetic():
    from analysis import cost as c
    per_req = c.gpu_seconds_per_request(0.5)           # 1 GPU at 0.5 req/s -> 1 / 0.5 = 2 GPU-seconds per request
    assert per_req == pytest.approx(2.0)
    assert c.gpu_seconds_per_request(2.0, num_gpus=2, utilisation=0.5) == pytest.approx(2.0)   # 2 / 2 / 0.5
    assert c.blended_gpu_hours_per_million_tokens(3.6, 400, 100) == pytest.approx(2.0)         # 3.6 / 3600 / 500 * 1e6
    api = c.api_gpu_seconds_per_request(400, 100, 1.0, 3.0)   # (400*1.0 + 100*3.0) / 1e6 GPU-h = 7e-4 h = 2.52 s
    assert api == pytest.approx(2.52)
    assert c.breakeven_utilisation(1.26, 2.52) == pytest.approx(0.5)
    for bad in (lambda: c.gpu_seconds_per_request(0, 1), lambda: c.gpu_seconds_per_request(1, 1, utilisation=1.5),
                lambda: c.gpu_seconds_per_request(1, 0), lambda: c.blended_gpu_hours_per_million_tokens(1, 0, 0),
                lambda: c.api_gpu_seconds_per_request(1, 1, -1, 1), lambda: c.breakeven_utilisation(1, 0)):
        with pytest.raises(ValueError):
            bad()


# ---- the optional quality-equivalence comparator --------------------------------------------
def _rows(texts: dict) -> dict:
    return {k: {"kind": k.split("-")[0], "text": v} for k, v in texts.items()}


def test_quality_comparator_scores_agreement():
    base = _rows({"fact-1": "the capital is Canberra", "code-1": "def f(x): return x + 1", "copy-1": "a b c d"})
    same = ce.compare(base, base)
    assert same["exact"] == 1.0 and same["prefix"] == 1.0 and same["diverge_median_word"] is None
    cand = _rows({"fact-1": "the capital is Canberra", "code-1": "def f(x): return x + 2", "copy-1": "z b c d"})
    r = ce.compare(base, cand)
    assert r["exact"] == pytest.approx(1 / 3)
    # code-1 differs at the last of 6 words (5 of 6 agree), copy-1 at the first of 4 (0 of 4 agree)
    assert r["prefix"] == pytest.approx((1.0 + 5 / 6 + 0.0) / 3)
    assert r["diverge_median_word"] == 2.5         # first differences at word 5 and word 0 -> median 2.5
    assert set(r["by_kind"]) == {"fact", "code", "copy"}
    assert r["worst"][0] == "copy-1"
    with pytest.raises(ValueError):
        ce.compare(base, _rows({"fact-1": "x"}))


def test_quality_cli_exit_codes(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = [{"id": f"fact-{i}", "kind": "fact", "text": f"answer {i} is here"} for i in range(4)]
    p1.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rows[0]["text"] = "answer 0 differs now"
    p2.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    argv = ["--baseline", str(p1), "--candidate", str(p2)]
    assert ce.main(argv + ["--min-exact", "0.7", "--min-prefix", "0.8"]) == 0
    assert ce.main(argv + ["--min-exact", "0.9", "--min-prefix", "0.8"]) == 1
    assert ce.main(["--baseline", str(p1), "--candidate", str(tmp_path / "missing.jsonl"),
                    "--min-exact", "0.5", "--min-prefix", "0.5"]) == 2


def test_the_prompt_set_is_well_formed():
    rows = [json.loads(line) for line in (ROOT / "quality" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 30 and len({r["id"] for r in rows}) == len(rows)
    assert {r["kind"] for r in rows} == {"fact", "math", "code", "copy", "reason"}
    assert all(r["prompt"].strip() for r in rows)


def test_run_prompts_records_outputs_and_refuses_engine_sim(tmp_path, monkeypatch):
    real_client = httpx.Client

    def make(sim: bool):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/sim/info":
                return httpx.Response(200 if sim else 404, json={"engine_sim": True})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "fake-model"}]})
            body = json.loads(request.content)
            assert body["temperature"] == 0 and body["model"] == "fake-model"
            return httpx.Response(200, json={"choices": [{"text": " echo " + body["prompt"][:5], "finish_reason": "stop"}]})
        monkeypatch.setattr(run_prompts.httpx, "Client",
                            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    make(sim=True)
    assert run_prompts.main(["--name", "x", "--out-dir", str(tmp_path)]) == 2 and not list(tmp_path.glob("*.jsonl"))
    make(sim=False)
    assert run_prompts.main(["--name", "fake", "--out-dir", str(tmp_path)]) == 0
    rows = [json.loads(line) for line in (tmp_path / "fake.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 30 and rows[0]["text"].startswith(" echo ") and rows[0]["kind"]
    assert set(ce.load(tmp_path / "fake.jsonl")) == {r["id"] for r in rows}


# ---- gpu/ and k8s/ -----------------------------------------------------------------------------
def test_tp2_script_uses_the_documented_flag():
    text = (ROOT / "gpu" / "serve_tp2.sh").read_text(encoding="utf-8")
    assert "--tensor-parallel-size" in text and "vllm serve" in text and "set -euo pipefail" in text
    assert "parallelism_scaling" in text, "cite the docs page the flag comes from"


def test_gang_lab_manifest_is_well_formed_and_the_scheduler_model_predicts_the_deadlock():
    import predict
    r = predict.predict()
    assert r["gang_sizes"] == {"job-a": 4, "job-b": 4} and r["capacity"] == 6
    assert r["held"] == {"job-a": 3, "job-b": 3} and r["pending"] == ["a3", "b3"]
    assert r["deadlock"] and not r["complete_gangs"]
    # With 8 GPUs both gangs fit: the deadlock comes from the capacity, not from the manifest.
    assert not predict.predict(gpus_per_node=4)["deadlock"]
    assert "~1" in (ROOT / "k8s" / "gang_lab" / "fake-gpus.sh").read_text(encoding="utf-8")


# ---- the optional profiler script ---------------------------------------------------------------
def test_profile_script_runs_on_cpu(tmp_path):
    pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "profiling" / "profile_decode_step.py"), "--hidden", "128", "--layers", "2",
         "--batches", "1,4", "--repeats", "2", "--trace-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "aten::mm" in proc.stdout or "aten::linear" in proc.stdout
    trace = json.loads((tmp_path / "cpu_profile_batch1.json").read_text(encoding="utf-8"))
    assert trace.get("traceEvents")
