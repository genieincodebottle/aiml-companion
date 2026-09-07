"""Tests for the LangGraph pipeline structure."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.state import default_state
from src.guardrails.manager import GuardrailManager


class TestGraphRouting:
    """Test routing logic without actually invoking LLMs."""

    def test_route_after_fact_check_no_contradictions(self):
        from src.agents.graph import _route_after_fact_check
        state = default_state("Test")
        state["contradictions"] = []
        result = _route_after_fact_check(state)
        assert result == "synthesize_report"

    def test_route_after_fact_check_with_contradictions(self):
        from src.agents.graph import _route_after_fact_check
        state = default_state("Test")
        state["contradictions"] = [{"claim": "test", "status": "unresolved"}]
        result = _route_after_fact_check(state)
        assert result == "resolve_contradictions"

    def test_route_after_fact_check_resolved_contradictions(self):
        from src.agents.graph import _route_after_fact_check
        state = default_state("Test")
        state["contradictions"] = [{"claim": "test", "status": "resolved"}]
        result = _route_after_fact_check(state)
        assert result == "synthesize_report"

    def test_fan_out_returns_four_sends(self):
        from src.agents.graph import _fan_out_to_specialists
        state = default_state("Test")
        sends = _fan_out_to_specialists(state)
        assert len(sends) == 4

    def test_graph_compiles(self):
        from src.agents.graph import build_graph
        gm = GuardrailManager()
        app = build_graph(gm)
        # Should compile without error
        assert app is not None


class TestGuardedNode:
    """Test that guardrail-wrapped nodes block correctly."""

    def test_blocked_agent_returns_error(self):
        from src.agents.graph import _create_guarded_node

        gm = GuardrailManager(config={
            "max_agent_calls": 0,  # Block immediately
            "max_reasoning_iterations": 10,
            "max_execution_time_seconds": 300,
        })
        gm.start_pipeline()

        dummy_fn = lambda state: {"result": "should not reach here"}
        guarded = _create_guarded_node(dummy_fn, "test_agent", gm)

        state = default_state("Test")
        result = guarded(state)

        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "guardrail" in result["errors"][0].lower()
