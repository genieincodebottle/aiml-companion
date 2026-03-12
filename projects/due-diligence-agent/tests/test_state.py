"""Tests for pipeline state schema."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.state import default_state, DueDiligenceState


class TestDefaultState:
    def test_creates_valid_state(self):
        state = default_state("Tesla")
        assert state["company_name"] == "Tesla"
        assert state["status"] == "planning"
        assert state["analysis_depth"] == "standard"

    def test_custom_query(self):
        state = default_state("Stripe", query="Focus on fintech competition")
        assert "fintech" in state["query"]

    def test_custom_depth(self):
        state = default_state("OpenAI", depth="deep")
        assert state["analysis_depth"] == "deep"

    def test_all_lists_empty(self):
        state = default_state("Test")
        assert state["financial_findings"] == []
        assert state["news_findings"] == []
        assert state["competitive_findings"] == []
        assert state["risk_findings"] == []
        assert state["errors"] == []
        assert state["pipeline_trace"] == []

    def test_default_query_includes_company(self):
        state = default_state("Databricks")
        assert "Databricks" in state["query"]

    def test_numeric_defaults(self):
        state = default_state("Test")
        assert state["total_tokens_used"] == 0
        assert state["total_cost_usd"] == 0.0
        assert state["overall_confidence"] == 0.0
