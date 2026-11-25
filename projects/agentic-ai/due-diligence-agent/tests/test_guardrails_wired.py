"""Guardrails that used to be defined and never applied.

- post_check was handed 0 tokens and $0.00 for every node, so the token and cost
  budgets could never fill and pre_check could never trip them.
- mask_pii existed and nothing called it: PII was flagged in a warning and then
  printed in the report anyway.
- confidence_threshold was read into a local variable and dropped.

These tests drive the real guarded nodes and the real (offline) graph.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import _create_guarded_node, run_pipeline
from src.guardrails.manager import GuardrailManager
from src.llm import invoke_with_tracking, token_tracker, usage_scope
from src.models.state import default_state

CONFIG = {
    "max_agent_calls": 30,
    "max_reasoning_iterations": 10,
    "max_execution_time_seconds": 300,
    "confidence_threshold": 0.6,
    "pii_detection_enabled": True,
    "source_grounding_required": True,
}


def _manager(**overrides) -> GuardrailManager:
    gm = GuardrailManager(config={**CONFIG, **overrides})
    gm.start_pipeline()
    return gm


class _Msg:
    def __init__(self, input_tokens, output_tokens):
        self.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens}


# === Budgets are charged with real usage ===

def test_usage_is_read_from_usage_metadata_for_structured_calls():
    class Chain:
        def invoke(self, prompt):
            return {"raw": _Msg(1000, 500), "parsed": "PARSED", "parsing_error": None}

    with usage_scope() as usage:
        out = invoke_with_tracking(Chain(), "p", "t")
    assert out == "PARSED"
    assert usage["total_tokens"] == 1500
    assert usage["cost_usd"] == pytest.approx(1.0 * 0.0001 + 0.5 * 0.0004)


def test_a_parse_failure_still_raises(monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda _s: None)  # skip retry backoff

    class Chain:
        def invoke(self, prompt):
            return {"raw": _Msg(1, 1), "parsed": None, "parsing_error": ValueError("bad")}

    with pytest.raises(ValueError, match="failed to parse"):
        invoke_with_tracking(Chain(), "p", "t")


def test_guarded_node_charges_the_real_usage_to_the_budget():
    class Chain:
        def invoke(self, prompt):
            return {"raw": _Msg(2000, 1000), "parsed": "x", "parsing_error": None}

    def agent(state):
        invoke_with_tracking(Chain(), "p", "agent")
        invoke_with_tracking(Chain(), "p", "agent")
        return {}

    gm = _manager()
    _create_guarded_node(agent, "agent", gm)(default_state("Acme"))
    assert gm.state.total_tokens_used == 6000
    assert gm.state.total_cost_usd > 0


def test_pipeline_fills_the_token_and_cost_budgets(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    before = token_tracker.total_tokens
    gm = _manager()
    result = run_pipeline("Acme", guardrail_mgr=gm)
    spent = token_tracker.total_tokens - before
    assert spent > 0
    assert gm.state.total_tokens_used == spent
    assert result["_budget_status"]["tokens_used"] == spent
    assert result["_budget_status"]["cost_usd"] > 0


def test_an_exhausted_token_budget_now_blocks_later_agents(monkeypatch):
    """With 0 tokens recorded this could never happen."""
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    gm = _manager()
    gm.budget = {**gm.budget, "max_total_tokens": 1}
    result = run_pipeline("Acme", guardrail_mgr=gm)
    assert any("Token budget exhausted" in e for e in result["errors"])


# === PII is masked, not just flagged ===

def test_pii_is_masked_in_agent_output_but_urls_survive():
    url = "https://www.sec.gov/Archives/edgar/data/1318605/0001564590-21-004599.htm"

    def agent(state):
        return {
            "risk_findings": [{"title": "t", "detail": "Reach john@test.com or 555-123-4567.",
                               "sources": [url], "confidence": 0.9}],
            "final_report": f"SSN 123-45-6789, see {url}",
        }

    gm = _manager()
    out = _create_guarded_node(agent, "agent", gm)(default_state("Acme"))
    finding = out["risk_findings"][0]
    assert "john@test.com" not in finding["detail"]
    assert "[EMAIL_REDACTED]" in finding["detail"] and "[PHONE_REDACTED]" in finding["detail"]
    assert "123-45-6789" not in out["final_report"]
    assert finding["sources"] == [url]
    assert url in out["final_report"]
    assert any("PII detected" in w for w in out["warnings"])


def test_pii_masking_respects_the_config_switch():
    def agent(state):
        return {"final_report": "mail john@test.com"}

    gm = _manager(pii_detection_enabled=False)
    out = _create_guarded_node(agent, "agent", gm)(default_state("Acme"))
    assert out["final_report"] == "mail john@test.com"


def test_masking_runs_on_every_node_of_the_real_graph(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    calls = {"mask": 0, "confidence": 0}
    real_mask, real_conf = GuardrailManager.mask_result, GuardrailManager.apply_confidence_threshold

    def mask(self, result):
        calls["mask"] += 1
        return real_mask(self, result)

    def conf(self, agent_name, result):
        calls["confidence"] += 1
        return real_conf(self, agent_name, result)

    monkeypatch.setattr(GuardrailManager, "mask_result", mask)
    monkeypatch.setattr(GuardrailManager, "apply_confidence_threshold", conf)
    run_pipeline("Acme", guardrail_mgr=_manager())
    assert calls["mask"] >= 6 and calls["confidence"] == calls["mask"]


# === Confidence threshold flags weak findings ===

def _findings_agent(state):
    return {"financial_findings": [
        {"title": "Strong", "detail": "d", "confidence": 0.9},
        {"title": "Weak", "detail": "d", "confidence": 0.4},
    ]}


def test_findings_below_the_threshold_are_flagged_not_dropped():
    gm = _manager()
    out = _create_guarded_node(_findings_agent, "financial_analyst", gm)(default_state("Acme"))
    by_title = {f["title"]: f for f in out["financial_findings"]}
    assert len(by_title) == 2, "a weak finding must stay in the report"
    assert by_title["Weak"].get("low_confidence") is True
    assert "low_confidence" not in by_title["Strong"]
    assert any("confidence threshold" in w and "Weak" in w for w in out["warnings"])


def test_the_threshold_is_configurable():
    gm = _manager(confidence_threshold=0.95)
    out = _create_guarded_node(_findings_agent, "financial_analyst", gm)(default_state("Acme"))
    assert all(f.get("low_confidence") for f in out["financial_findings"])

    gm = _manager(confidence_threshold=0.1)
    out = _create_guarded_node(_findings_agent, "financial_analyst", gm)(default_state("Acme"))
    assert not any(f.get("low_confidence") for f in out["financial_findings"])


def test_the_report_marks_low_confidence_findings():
    from src.agents.lead_analyst import _build_markdown_report
    from src.models.schemas import ExecutiveSummary

    state = default_state("Acme")
    state["financial_findings"] = [{"title": "Weak", "detail": "d", "confidence": 0.4,
                                    "low_confidence": True}]
    summary = ExecutiveSummary(
        company_name="Acme",
        one_line_verdict="v", overall_risk_rating="moderate", overall_confidence=0.7,
        recommendation="proceed_with_caution", key_strengths=[], key_risks=[],
        key_uncertainties=[], action_items=[],
    )
    assert "LOW CONFIDENCE" in _build_markdown_report(state, summary)
