"""Config keys that were read by no code, now enforced.

budget.max_tokens_per_agent, pipeline.timeout_per_agent_seconds,
pipeline.max_retries_per_agent and guardrails.hallucination_check_enabled, plus
real per-agent trace tokens and repeat-entry loop detection.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.agents.graph as graph_mod
from src.agents.graph import _create_guarded_node, build_graph, run_pipeline
from src.guardrails.manager import GuardrailManager
from src.llm import AgentLimitExceeded, invoke_with_tracking, token_tracker
from src.models.state import default_state

CONFIG = {
    "max_agent_calls": 30,
    "max_reasoning_iterations": 10,
    "max_execution_time_seconds": 300,
    "confidence_threshold": 0.6,
    "pii_detection_enabled": True,
    "source_grounding_required": True,
}


def _manager(budget=None, **overrides) -> GuardrailManager:
    gm = GuardrailManager(config={**CONFIG, **overrides})
    if budget:
        gm.budget = {**gm.budget, **budget}
    gm.start_pipeline()
    return gm


class _Msg:
    def __init__(self, i, o):
        self.usage_metadata = {"input_tokens": i, "output_tokens": o}


class _Chain:
    """One call = 600 tokens."""
    def invoke(self, prompt):
        return {"raw": _Msg(400, 200), "parsed": "ok", "parsing_error": None}


def _pipeline_cfg(monkeypatch, **cfg):
    monkeypatch.setattr(graph_mod, "get_pipeline_config", lambda: cfg)


# === (1) per-agent token ceiling ===

def test_agent_is_stopped_before_the_call_that_would_exceed_its_ceiling(monkeypatch):
    _pipeline_cfg(monkeypatch)
    calls = []

    def agent(state):
        try:
            for _ in range(5):
                invoke_with_tracking(_Chain(), "p", "agent")
                calls.append(1)
        except AgentLimitExceeded as e:   # what a real agent's try/except does
            return {"errors": [str(e)]}
        return {}

    gm = _manager(budget={"max_tokens_per_agent": 1000})
    out = _create_guarded_node(agent, "agent", gm)(default_state("Acme"))
    assert len(calls) == 2, "third call must be refused once 1200 >= 1000"
    assert "token ceiling" in out["errors"][0]
    assert any("per-agent ceiling" in w for w in out["warnings"])
    assert any(v["type"] == "agent_token_ceiling" for v in gm.get_violations())


def test_pre_check_blocks_an_agent_that_already_spent_its_ceiling():
    gm = _manager(budget={"max_tokens_per_agent": 1000})
    gm.post_check("agent", "text https://x.test", tokens_used=1500)
    result = gm.pre_check("agent")
    assert result["passed"] is False and "token ceiling" in result["reason"]
    assert gm.pre_check("other_agent")["passed"] is True


# === (2) per-agent timeout ===

def test_a_slow_agent_times_out_with_a_warning_and_is_told_to_stop(monkeypatch):
    _pipeline_cfg(monkeypatch, timeout_per_agent_seconds=0.05)
    later = []

    def slow_agent(state):
        time.sleep(0.25)
        try:
            invoke_with_tracking(_Chain(), "p", "slow")
        except AgentLimitExceeded as e:
            later.append(e)
        return {"final_report": "too late"}

    gm = _manager()
    out = _create_guarded_node(slow_agent, "slow", gm)(default_state("Acme"))
    assert "timed out" in out["errors"][0]
    assert "timed out after 0.05s" in out["warnings"][0]
    assert out["pipeline_trace"][0]["action"] == "timeout"
    assert "final_report" not in out
    time.sleep(0.4)  # the abandoned thread wakes up and finds itself cancelled
    assert later and "cancelled" in str(later[0])


def test_a_fast_agent_is_unaffected_by_the_timeout(monkeypatch):
    _pipeline_cfg(monkeypatch, timeout_per_agent_seconds=5)
    gm = _manager()
    out = _create_guarded_node(lambda s: {"final_report": "done"}, "fast", gm)(default_state("Acme"))
    assert out["final_report"] == "done"


# === (3) retries ===

class _Failing:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise RuntimeError("boom")


@pytest.mark.parametrize("per_agent, expected_attempts", [(0, 1), (1, 2), (2, 3), (9, 3)])
def test_max_retries_per_agent_caps_the_attempts(monkeypatch, per_agent, expected_attempts):
    monkeypatch.setattr("src.llm.time.sleep", lambda _s: None)
    monkeypatch.setattr("src.llm.get_pipeline_config", lambda: {"max_retries_per_agent": per_agent})
    chain = _Failing()
    with pytest.raises(RuntimeError, match="boom"):
        invoke_with_tracking(chain, "p", "agent")
    assert chain.calls == expected_attempts   # model.max_retries (3) is the upper bound


# === (4) hallucination check flag ===

def test_hallucination_check_flag_gates_the_fact_checker(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    on = run_pipeline("Acme", guardrail_mgr=_manager())
    assert on["fact_check_results"] and not any("Fact-checking disabled" in w for w in on["warnings"])

    off = run_pipeline("Acme", guardrail_mgr=_manager(hallucination_check_enabled=False))
    assert off["fact_check_results"] == []
    assert any("Fact-checking disabled" in w for w in off["warnings"])
    assert off["final_report"], "the pipeline still produces a report"


# === (5) real tokens on the trace ===

def test_trace_entries_carry_the_nodes_real_usage():
    def agent(state):
        invoke_with_tracking(_Chain(), "p", "agent")
        invoke_with_tracking(_Chain(), "p", "agent")
        return {"pipeline_trace": [{"agent": "agent", "tokens_used": 0, "cost_usd": 0.0}]}

    out = _create_guarded_node(agent, "agent", _manager())(default_state("Acme"))
    assert out["pipeline_trace"][0]["tokens_used"] == 1200
    assert out["pipeline_trace"][0]["cost_usd"] > 0


def test_pipeline_trace_tokens_add_up_to_what_was_spent(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    before = token_tracker.total_tokens
    result = run_pipeline("Acme", guardrail_mgr=_manager())
    spent = token_tracker.total_tokens - before
    assert spent > 0
    assert sum(t["tokens_used"] for t in result["pipeline_trace"]) == spent


# === (6) repeat-entry loop detection ===

def test_same_agent_with_an_identical_state_is_a_loop():
    gm = _manager()
    assert gm.pre_check("agent", "abc")["passed"] is True
    again = gm.pre_check("agent", "abc")
    assert again["passed"] is False and "identical state" in again["reason"]
    assert gm.pre_check("agent", "def")["passed"] is True      # state changed: progress
    assert gm.pre_check("other", "abc")["passed"] is True      # different agent
    assert gm.pre_check("agent")["passed"] is True             # no fingerprint: not checked


def test_guarded_node_blocks_a_second_entry_with_the_same_state(monkeypatch):
    _pipeline_cfg(monkeypatch)
    gm = _manager()
    node = _create_guarded_node(lambda s: {"final_report": "x"}, "agent", gm)
    state = default_state("Acme")
    assert "final_report" in node(state)
    blocked = node(state)
    assert "identical state" in blocked["errors"][0]
