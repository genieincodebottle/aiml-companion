"""Tests for the Research Agent."""

import pytest
from unittest.mock import patch, MagicMock

from agents.research_agent import run, SYSTEM_PROMPT


class TestResearchAgent:
    """Test the research agent's core functionality."""

    def test_empty_query_returns_error(self):
        """Agent should return error when no query is provided."""
        result = run({"query": ""})
        assert "Error" in result["result"]
        assert result["agent"] == "research"

    def test_missing_query_returns_error(self):
        """Agent should handle missing query key."""
        result = run({})
        assert "Error" in result["result"]

    @patch("agents.research_agent.web_search")
    @patch("agents.research_agent.call_llm")
    def test_successful_research(self, mock_llm, mock_search):
        """Agent should search web and synthesize findings."""
        mock_search.return_value = [
            {"title": "Test Result", "url": "https://example.com", "content": "Test content"}
        ]
        mock_llm.return_value = "## Key Findings\n- Test finding [Source 1]"

        result = run({"query": "test query"})

        assert result["agent"] == "research"
        assert "Key Findings" in result["result"]
        assert len(result["sources"]) == 1
        mock_search.assert_called_once()
        mock_llm.assert_called_once()

    @patch("agents.research_agent.web_search")
    @patch("agents.research_agent.call_llm")
    def test_research_with_context(self, mock_llm, mock_search):
        """Agent should do refined search when context is provided."""
        mock_search.return_value = [
            {"title": "Result", "url": "https://example.com", "content": "Content"}
        ]
        mock_llm.return_value = "Research summary"

        result = run({"query": "AI agents", "context": "focus on AWS"})

        # Should search twice: once for query, once for refined query
        assert mock_search.call_count == 2
        assert result["agent"] == "research"

    def test_system_prompt_is_defined(self):
        """System prompt should exist and contain key instructions."""
        assert "research" in SYSTEM_PROMPT.lower()
        assert "source" in SYSTEM_PROMPT.lower()
