"""Offline mode is a product feature, so it is tested like one.

The point of these tests is that a fixture drifting from its schema must FAIL
here. An earlier version of the offline provider fell back to
`model_construct` on a validation error: the pipeline then ran to completion
while three of the four agents degraded to "analysis incomplete", and the
report still looked plausible.
"""
from __future__ import annotations

import os

import pytest

from src.llm_offline import (OfflineFixtureError, OfflineChatModel, _build,
                             _fixtures)
from src.models import schemas as S


FIXTURE_SCHEMAS = [
    "ResearchPlan", "FinancialAnalysis", "NewsSentimentAnalysis",
    "CompetitiveAnalysis", "RiskAssessment", "FactCheckReport",
    "ExecutiveSummary", "ConflictResolution",
]


@pytest.mark.parametrize("name", FIXTURE_SCHEMAS)
def test_every_fixture_validates_against_its_schema(name):
    """The gate. A schema change breaks this rather than the demo output."""
    schema = getattr(S, name)
    data = _fixtures("Northwind Robotics", 0)[name]
    obj = _build(schema, data)
    assert isinstance(obj, schema)


def test_a_broken_fixture_raises_rather_than_degrading():
    with pytest.raises(OfflineFixtureError):
        _build(S.FinancialAnalysis, {"company_name": 123, "financial_health_rating": []})


def test_the_fixtures_actually_disagree():
    """A demo where every agent agrees exercises the graph and teaches nothing.

    The financial analyst reports accelerating revenue; the news agent reports
    guidance withdrawn in the same quarter. The fact checker has to flag it and
    the lead analyst's debate has to run.
    """
    fx = _fixtures("Northwind Robotics", 0)
    assert "accelerat" in fx["FinancialAnalysis"]["revenue_analysis"]
    assert "withdraw" in fx["NewsSentimentAnalysis"]["key_events"][0]["headline"].lower()
    contradictions = fx["FactCheckReport"]["cross_agent_contradictions"]
    assert len(contradictions) >= 2
    assert any("financial_analyst" in c and "news_sentiment" in c for c in contradictions)


def test_at_least_one_claim_is_left_unverifiable():
    """An analyst that verifies everything has not been tested on anything."""
    v = _fixtures("X", 0)["FactCheckReport"]["verifications"]
    assert any(x["verification_status"] == "unverifiable" for x in v)


def test_structured_output_returns_the_requested_schema():
    llm = OfflineChatModel()
    out = llm.with_structured_output(S.FinancialAnalysis).invoke("Company: Acme")
    assert isinstance(out, S.FinancialAnalysis)
    assert out.key_metrics


def test_offline_search_needs_no_network():
    from src.tools.search_offline import OfflineSearch

    rows = OfflineSearch().search("anything", 3)
    assert len(rows) == 3
    # One deliberately weak source, so an agent has to weigh what it trusts.
    assert min(r.relevance_score for r in rows) < 0.5
    assert all(r.url.startswith("offline://") for r in rows)


def test_search_tool_delegates_when_offline(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    from src.tools.search import WebSearchTool

    tool = WebSearchTool()
    assert tool._offline is not None
    assert all(r.url.startswith("offline://") for r in tool.search("q"))


def test_offline_provider_is_registered():
    from src.llm import _PROVIDER_FACTORY

    assert "offline" in _PROVIDER_FACTORY
