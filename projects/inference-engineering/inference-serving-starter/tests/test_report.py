"""GATE 6: the one-page report. Copy report/REPORT_TEMPLATE.md to report/REPORT.md and finish it.

The test cannot judge whether your reasoning is good. It checks that the report is finished,
that it quotes the numbers your own analysis produced (results/decision.json), and that the
plot it shows exists. Read your draft against the "WEAK:" notes in the template.
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
            "Cost per million tokens", "Fleet-size recommendation", "Threats to validity"]


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
    assert 350 <= words <= 1100, f"{words} words: a one-page report is 350 to 1100"


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


def test_report_is_honest_about_the_simulator(text):
    low = text.lower()
    if "engine_sim" in low:
        assert "model" in low and ("not a gpu" in low or "simulat" in low or "simplif" in low), \
            "if you used engine_sim, say it is a model and what that means for your numbers"
    threats = text.split("Threats to validity")[-1]
    assert len(re.findall(r"^\s*[-*]\s+\S", threats, re.M)) >= 3, "list at least 3 concrete threats to validity"
