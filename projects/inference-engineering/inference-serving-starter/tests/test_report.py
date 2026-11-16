"""GATE 6: the one-page report. Copy report/REPORT_TEMPLATE.md to report/REPORT.md and finish it.

The test cannot judge whether your reasoning is good. It checks that the report is finished,
that it quotes the numbers your own analysis produced (results/decision.json), that the plot it
shows exists, and that one sweep in results/ came from a real `vllm serve` (gpu/README.md), with
the GPU, the vLLM version and the cross-check written up. It also checks that the report
quotes the second SLO row, the self-host-versus-API break-even and the open-loop p99 TTFT.
Read your draft against the "WEAK:" notes in the template.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "REPORT.md"
DECISION = ROOT / "results" / "decision.json"
SECTIONS = ["Setup", "Method", "Results", "Quantisation comparison", "Speculative decoding",
            "Cost per million tokens", "Fleet-size recommendation", "Measured on a real GPU",
            "Threats to validity"]
GPU_NAMES = r"\b(T4|L4|L40S?|A10G?|A100|H100|H200|B200|B300|GB200|RTX\s?\w+|MI\d{3}X?)\b"


def real_server_runs() -> list[Path]:
    """Sweeps whose meta.engine is null: bench.cli found no /sim/info, so the server was real."""
    runs = []
    for p in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        meta = d.get("meta") if isinstance(d, dict) else None
        if isinstance(meta, dict) and "engine" in meta and meta["engine"] is None and d.get("levels"):
            runs.append(p)
    return runs


@pytest.fixture(scope="module")
def text() -> str:
    if not REPORT.exists():
        pytest.fail("report/REPORT.md does not exist: copy report/REPORT_TEMPLATE.md and finish it")
    return REPORT.read_text(encoding="utf-8")


def test_no_template_left_in_the_report(text):
    assert "<<" not in text and ">>" not in text, "unreplaced <<marker>>"
    assert "WEAK:" not in text, "delete the WEAK notes"
    assert "TODO" not in text


def test_all_sections_present_and_length_is_one_page(text):
    for s in SECTIONS:
        assert re.search(rf"^##\s*\d*\.?\s*{re.escape(s)}", text, re.M | re.I), f"missing section: {s}"
    words = len(re.sub(r"<!--.*?-->|!\[.*?\]\(.*?\)", "", text, flags=re.S).split())
    assert 350 <= words <= 1250, f"{words} words: a one-page report is 350 to 1250"


def test_the_plot_exists(text):
    imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert imgs, "no knee plot embedded"
    for i in imgs:
        assert (REPORT.parent / i).resolve().exists(), f"image not found: {i}"


def test_report_quotes_the_decision_numbers(text):
    if not DECISION.exists():
        pytest.fail("results/decision.json missing: run python -m analysis.decide ...")
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    assert d["gpu_hourly_usd"] >= 0 and d["price_source"].strip(), "price must be your input with a source"
    cost = d["cost_per_million_output_tokens_usd"]
    candidates = {f"{cost:.2f}", f"{cost:.3f}", f"{cost:.1f}", f"{cost:.4f}"}
    assert any(c in text for c in candidates), f"report must state the cost per million tokens (about ${cost:.3f})"
    assert re.search(rf"\b{d['replicas']}\s+replicas?\b", text, re.I), f"report must state {d['replicas']} replicas"
    assert str(d["operating_concurrency"]) in text
    assert d["price_source"].split(",")[0][:20] in text or "price" in text.lower()


def _variants(v: float, extra: tuple = ()) -> set:
    return {f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.0f}", *extra}


def test_report_has_a_second_slo_row(text):
    if not DECISION.exists():
        pytest.fail("results/decision.json missing: run python -m analysis.decide ...")
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    second = d.get("second_slo")
    assert second, "decision.json has no second_slo: re-run python -m analysis.decide with the current analysis/decide.py"
    assert second["slo"] != d["slo"], "the second SLO must differ from the first"
    assert str(second["operating_concurrency"]) in text
    c2 = second["cost_per_million_output_tokens_usd"]
    assert any(v in text for v in (f"{c2:.2f}", f"{c2:.3f}", f"{c2:.1f}", f"{c2:.4f}")), (
        f"report must state the cost per million tokens under the second SLO (about ${c2:.3f})")
    assert f"{second['slo']['ttft_s']:g}" in text, "name the second SLO's TTFT limit in the table"


def test_report_compares_self_hosting_with_an_api(text):
    if not DECISION.exists():
        pytest.fail("results/decision.json missing: run python -m analysis.decide ...")
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    api = d.get("api") or {}
    assert api.get("source", "").strip() and api.get("input_usd_per_m", 0) >= 0 and api.get("output_usd_per_m", 0) >= 0, (
        "decision.json needs the API prices and the page and date you read them from")
    assert api.get("breakeven_utilisation") is not None, "API prices must be above zero"
    u = api["breakeven_utilisation"]
    low = text.lower()
    assert "api" in low and re.search(r"break(s)?[- ]?even", low), "the cost section needs a self-host vs API line with the break-even utilisation"
    assert any(v in text for v in _variants(u, (f"{u * 100:.0f}", f"{u * 100:.1f}"))), (
        f"quote the break-even utilisation (about {u:.2f}, or {u * 100:.0f} percent)")
    assert f"{d['mean_prompt_tokens']:.0f}" in text, "state the mean input tokens per request"


def test_report_quotes_the_open_loop_result(text):
    p = ROOT / "results" / "open_bf16.json"
    if not p.exists():
        pytest.fail("results/open_bf16.json is missing: run python scripts/run_experiments.py")
    from analysis.results_io import load_results
    from analysis.tables import percentile
    from bench.records import drop_warmup
    meta, levels = load_results(p)
    recs = [r for r in drop_warmup(levels[0]["records"], warmup_requests=meta["warmup_requests"]) if r.ok]
    p99 = percentile([r.ttft_s for r in recs], 99)
    assert re.search(r"open[- ]loop", text, re.I), "the report must discuss the open-loop run"
    assert any(v in text for v in _variants(p99)), f"quote the open-loop p99 TTFT (about {p99:.1f} s)"


def test_report_has_a_real_gpu_run(text):
    assert real_server_runs(), (
        "no sweep from a real GPU in results/: follow gpu/README.md and save one with "
        "python -m bench.cli sweep --base-url <your vllm serve> ... --out results/gpu_bf16.json")
    section = re.split(r"^##\s*\d*\.?\s*Measured on a real GPU", text, flags=re.M | re.I)[-1]
    section = re.split(r"^##\s", section, flags=re.M)[0]
    assert re.search(GPU_NAMES, section), "name the GPU you measured on"
    assert re.search(r"vllm\D{0,12}v?\d+\.\d+(\.\d+)?", section, re.I), "give the vLLM version you ran"
    assert "bench serve" in section.lower(), "quote the vllm bench serve cross-check"
    assert len(re.findall(r"^\s*\|", section, re.M)) >= 3, "add the sim-vs-real table (header, separator, rows)"


def test_report_is_honest_about_the_simulator(text):
    low = text.lower()
    if "engine_sim" in low:
        assert "model" in low and ("not a gpu" in low or "simulat" in low or "simplif" in low), \
            "if you used engine_sim, say it is a model and what that means for your numbers"
    threats = text.split("Threats to validity")[-1]
    assert len(re.findall(r"^\s*[-*]\s+\S", threats, re.M)) >= 3, "list at least 3 concrete threats to validity"
