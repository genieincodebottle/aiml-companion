"""Tests for the Orchestrator Agent."""

import json
import pytest
from unittest.mock import patch, MagicMock

from agents.orchestrator import create_plan, run, call_agent


class TestOrchestrator:
    """Test the orchestrator's planning and coordination."""

    @patch("agents.orchestrator.call_llm")
    def test_create_plan_parses_json(self, mock_llm):
        """Orchestrator should parse LLM plan response as JSON."""
        mock_llm.return_value = json.dumps({
            "plan": [
                {"agent": "research", "task": "search topic", "needs_previous": False},
                {"agent": "analyst", "task": "analyze data", "needs_previous": True},
                {"agent": "writer", "task": "write report", "needs_previous": True},
            ]
        })

        plan = create_plan("test query")

        assert len(plan) == 3
        assert plan[0]["agent"] == "research"
        assert plan[1]["agent"] == "analyst"
        assert plan[2]["agent"] == "writer"

    @patch("agents.orchestrator.call_llm")
    def test_create_plan_handles_markdown_wrapped_json(self, mock_llm):
        """Orchestrator should handle JSON wrapped in markdown code blocks."""
        mock_llm.return_value = '```json\n{"plan": [{"agent": "research", "task": "test", "needs_previous": false}]}\n```'

        plan = create_plan("test query")
        assert len(plan) == 1

    @patch("agents.orchestrator.call_llm")
    def test_create_plan_fallback_on_invalid_json(self, mock_llm):
        """Orchestrator should use default plan when LLM returns invalid JSON."""
        mock_llm.return_value = "This is not JSON at all."

        plan = create_plan("test query")

        # Should fall back to default 3-step plan
        assert len(plan) == 3
        assert plan[0]["agent"] == "research"

    @patch("agents.orchestrator.call_agent")
    @patch("agents.orchestrator.create_plan")
    def test_run_executes_all_steps(self, mock_plan, mock_agent):
        """Orchestrator should execute all steps in the plan."""
        mock_plan.return_value = [
            {"agent": "research", "task": "search", "needs_previous": False},
            {"agent": "analyst", "task": "analyze", "needs_previous": True},
            {"agent": "writer", "task": "write", "needs_previous": True},
        ]
        mock_agent.return_value = {"result": "Agent output", "agent": "test"}

        result = run("test query")

        assert len(result["steps"]) == 3
        assert result["query"] == "test query"
        assert result["final_report"] == "Agent output"
        assert result["total_time"] >= 0
        assert mock_agent.call_count == 3

    @patch("agents.orchestrator.call_agent")
    @patch("agents.orchestrator.create_plan")
    def test_run_passes_context_between_steps(self, mock_plan, mock_agent):
        """Research output should be passed to analyst, both to writer."""
        mock_plan.return_value = [
            {"agent": "research", "task": "search", "needs_previous": False},
            {"agent": "analyst", "task": "analyze", "needs_previous": True},
            {"agent": "writer", "task": "write", "needs_previous": True},
        ]

        call_count = [0]
        def side_effect(agent_name, input_data):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"result": "Research findings", "agent": "research"}
            elif call_count[0] == 2:
                # Analyst should receive research
                assert "Research findings" in input_data.get("research", "")
                return {"result": "Analysis results", "agent": "analyst"}
            else:
                # Writer should receive both
                assert "Research findings" in input_data.get("research", "")
                assert "Analysis results" in input_data.get("analysis", "")
                return {"result": "Final report", "agent": "writer"}

        mock_agent.side_effect = side_effect

        result = run("test query")
        assert result["final_report"] == "Final report"

    def test_call_agent_local_unknown_agent(self):
        """Should handle unknown agent names gracefully."""
        from agents.orchestrator import _call_agent_local

        result = _call_agent_local("nonexistent", {"query": "test"})
        assert "Error" in result["result"]
