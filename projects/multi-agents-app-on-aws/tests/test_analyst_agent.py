"""Tests for the Analyst Agent."""

import pytest
from unittest.mock import patch

from agents.analyst_agent import run, SYSTEM_PROMPT


class TestAnalystAgent:
    """Test the analyst agent's core functionality."""

    def test_no_research_returns_error(self):
        """Agent should return error when no research data is provided."""
        result = run({"query": "test", "research": ""})
        assert "Error" in result["result"]
        assert result["agent"] == "analyst"

    @patch("agents.analyst_agent.call_llm")
    def test_successful_analysis(self, mock_llm):
        """Agent should analyze research and return insights."""
        mock_llm.return_value = "## Analysis Summary\nKey insight here"

        result = run({
            "query": "Compare frameworks",
            "research": "LangGraph has 15k stars. CrewAI has 20k stars.",
        })

        assert result["agent"] == "analyst"
        assert "Analysis" in result["result"]
        mock_llm.assert_called_once()

    @patch("agents.analyst_agent.call_llm")
    def test_analysis_with_focus_areas(self, mock_llm):
        """Agent should include focus areas in its analysis."""
        mock_llm.return_value = "Focused analysis"

        run({
            "query": "Compare frameworks",
            "research": "Some research data",
            "focus_areas": "pricing and scalability",
        })

        # Check that focus areas were passed to the LLM
        call_args = mock_llm.call_args
        assert "pricing and scalability" in call_args.kwargs.get("user_message", "") or \
               "pricing and scalability" in str(call_args)

    def test_system_prompt_is_defined(self):
        """System prompt should contain analysis instructions."""
        assert "analyst" in SYSTEM_PROMPT.lower() or "analyz" in SYSTEM_PROMPT.lower()
        assert "comparison" in SYSTEM_PROMPT.lower() or "compar" in SYSTEM_PROMPT.lower()
