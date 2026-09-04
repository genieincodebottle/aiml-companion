"""A deterministic LLM provider that needs no API key and no network.

Why this exists
---------------
Every other provider in `src/llm.py` needs a key. That meant a reader could
clone the repository, run 77 passing tests, and still never see the pipeline
execute. Running it is the point.

What it is not
--------------
It is not a mock that returns empty objects. Each agent gets a schema-valid
response built from a fixture, and the fixtures deliberately DISAGREE:

- the financial analyst reports revenue growth accelerating
- the news agent reports guidance withdrawn in the same quarter

Those cannot both be comfortable, so the fact checker flags a cross-agent
contradiction and the lead analyst's debate runs. A fixture set where every
agent agreed would exercise the graph and demonstrate nothing.

The seed is the company name, so two runs of the same company are identical
and a run of a different company differs.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

OFFLINE_NOTE = "offline fixture, not a live model call"


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _company_from(prompt: str) -> str:
    """Best effort. The prompt carries the company name; the fixtures only use
    it for flavour and for the seed, so a miss is harmless."""
    for line in str(prompt).splitlines():
        if "Company:" in line:
            return line.split("Company:", 1)[1].strip() or "the company"
    return "the company"


# ---------------------------------------------------------------------------
# Fixtures, keyed by schema class name.
#
# The two marked CONTRADICTION are the reason this file is not a stub.
# ---------------------------------------------------------------------------

def _fixtures(company: str, rng: int) -> dict[str, dict]:
    """Field names here are checked against src/models/schemas.py by
    tests/test_offline_mode.py. A schema change must break that test rather
    than quietly turn an agent's output into "analysis incomplete"."""
    return {
        "ResearchPlan": {
            "company_summary": f"{company} is the analysis target. Offline fixture data.",
            "sub_tasks": [
                {"agent": "financial_analyst", "focus": "revenue and margin trend",
                 "priority": "high",
                 "key_questions": ["Is revenue growth accelerating?",
                                   "Are margins holding?"]},
                {"agent": "news_sentiment", "focus": "last two quarters of coverage",
                 "priority": "high",
                 "key_questions": ["Has guidance changed?"]},
                {"agent": "competitive_intel", "focus": "share shift",
                 "priority": "medium", "key_questions": ["Who is gaining?"]},
                {"agent": "risk_assessor", "focus": "regulatory and concentration",
                 "priority": "medium", "key_questions": ["Any single-customer risk?"]},
            ],
            "focus_areas": ["revenue quality", "guidance credibility", "concentration"],
            "risk_hypothesis": "Reported growth may not survive the guidance change.",
        },

        # CONTRADICTION, half one. Growth is accelerating.
        "FinancialAnalysis": {
            "company_name": company,
            "financial_health_rating": "moderate",
            "key_metrics": [
                {"metric_name": "Revenue", "value": "$4.2B", "trend": "increasing",
                 "assessment": "positive", "source": "https://offline.example/filings/q3"},
                {"metric_name": "Gross Margin", "value": "41.3%", "trend": "stable",
                 "assessment": "neutral", "source": "https://offline.example/filings/q3"},
                {"metric_name": "Free Cash Flow", "value": "$310M", "trend": "decreasing",
                 "assessment": "concerning", "source": "https://offline.example/filings/q3"},
            ],
            "revenue_analysis": (
                "Revenue growth accelerated to 18 percent year on year in the most "
                "recent quarter, the third consecutive quarter of acceleration."),
            "profitability_analysis": (
                "Margins are flat while revenue grows, so operating leverage is not appearing."),
            "cash_flow_notes": (
                "Free cash flow fell from $402M to $310M despite the revenue growth, "
                "which is the thread worth pulling."),
            "red_flags": ["Free cash flow diverging from reported revenue"],
            "green_flags": ["Third consecutive quarter of revenue acceleration"],
            "data_gaps": ["No segment breakdown available offline"],
            "sources": ["https://offline.example/filings/q3"],
        },

        # CONTRADICTION, half two. Guidance withdrawn in the same quarter.
        "NewsSentimentAnalysis": {
            "company_name": company,
            "overall_sentiment": "negative",
            "sentiment_trend": "deteriorating",
            "key_events": [
                {"date": "most recent quarter",
                 "headline": f"{company} withdraws full-year guidance",
                 "sentiment": "negative", "impact": "high", "source": "https://offline.example/news/1",
                 "summary": "Management withdrew guidance citing demand visibility."},
                {"date": "most recent quarter",
                 "headline": "Two senior finance departures",
                 "sentiment": "negative", "impact": "medium", "source": "https://offline.example/news/2",
                 "summary": "The CFO and a controller left within six weeks."},
            ],
            "public_perception": (
                "Coverage is dominated by the withdrawn guidance, which sits badly "
                "against the reported acceleration in revenue."),
            "media_coverage_volume": "elevated",
            "social_media_notes": "Retail sentiment turned after the guidance withdrawal.",
            "potential_concerns": ["Guidance credibility", "Finance team turnover"],
            "sources": ["https://offline.example/news/1", "https://offline.example/news/2"],
        },

        "CompetitiveAnalysis": {
            "company_name": company,
            "industry": "Industrial automation",
            "market_position": "challenger",
            "competitors": [
                {"name": "Incumbent A", "market_position": "leader",
                 "key_strengths": ["Enterprise distribution", "Installed base"],
                 "key_weaknesses": ["Slow release cadence"],
                 "estimated_market_share": "38%"},
                {"name": "Startup B", "market_position": "niche",
                 "key_strengths": ["Price"], "key_weaknesses": ["No service network"],
                 "estimated_market_share": "4%"},
            ],
            "competitive_advantages": ["Service network", "Mid-market pricing"],
            "competitive_risks": ["Incumbent bundling", "Price pressure from below"],
            "market_trends": ["Consolidation among integrators"],
            "differentiation_summary": (
                "The moat is distribution rather than technology, which is weaker "
                "than it looks when an incumbent decides to bundle."),
            "sources": ["https://offline.example/market/1"],
        },

        "RiskAssessment": {
            "company_name": company,
            "overall_risk_level": "high",
            "risks": [
                {"risk_type": "financial", "title": "Cash conversion",
                 "description": "Free cash flow is diverging from reported revenue.",
                 "severity": "high", "likelihood": "likely",
                 "mitigation": "Reconcile revenue recognition against collections.",
                 "source": "https://offline.example/filings/q3"},
                {"risk_type": "governance", "title": "Guidance and turnover",
                 "description": "Guidance withdrawn and two finance departures in one quarter.",
                 "severity": "high", "likelihood": "certain",
                 "mitigation": "Meet the incoming finance leadership.",
                 "source": "https://offline.example/news/2"},
                {"risk_type": "concentration", "title": "Customer concentration",
                 "description": "A top customer may exceed 20 percent of revenue.",
                 "severity": "medium", "likelihood": "possible",
                 "mitigation": "Obtain segment disclosure.",
                 "source": "https://offline.example/blog/1"},
            ],
            "regulatory_environment": "No open actions found in offline data.",
            "legal_history": "Nothing material in the offline fixture.",
            "esg_concerns": "Not assessed offline.",
            "risk_summary": (
                "Two high-severity risks land in the same quarter, which is what "
                "moves the overall rating rather than either one alone."),
            "sources": ["https://offline.example/filings/q3", "https://offline.example/news/2"],
        },

        "FactCheckReport": {
            "total_claims_checked": 7,
            "verified_count": 4,
            "contradicted_count": 2,
            "unverifiable_count": 1,
            "verifications": [
                {"claim": "Revenue growth accelerated to 18 percent",
                 "source_agent": "financial_analyst",
                 "verification_status": "contradicted", "confidence": 0.55,
                 "evidence": "Guidance was withdrawn in the same quarter.",
                 "contradicts_agent": "news_sentiment",
                 "source": "https://offline.example/news/1"},
                {"claim": "A top customer may exceed 20 percent of revenue",
                 "source_agent": "risk_assessor",
                 "verification_status": "unverifiable", "confidence": 0.2,
                 "evidence": "Only an unsourced blog post supports it.",
                 "source": "https://offline.example/blog/1"},
            ],
            "cross_agent_contradictions": [
                "financial_analyst reports accelerating revenue growth while "
                "news_sentiment reports withdrawn full-year guidance in the same quarter",
                "financial_analyst rates health moderate while risk_assessor rates "
                "overall risk high",
            ],
            "overall_reliability": "moderate",
            "notes": OFFLINE_NOTE,
        },

        "ExecutiveSummary": {
            "company_name": company,
            "one_line_verdict": (
                "Real but low-quality growth, and the disagreement between the "
                "numbers and the guidance is the finding."),
            "overall_risk_rating": "high",
            "overall_confidence": 0.68,
            "key_strengths": [
                "Third consecutive quarter of revenue acceleration",
                "Service network the incumbent cannot cheaply copy",
            ],
            "key_risks": [
                "Free cash flow diverging from reported revenue",
                "Guidance withdrawn and two finance departures in one quarter",
            ],
            "key_uncertainties": [
                "Customer concentration could not be verified from available sources",
            ],
            "recommendation": "proceed_with_conditions",
            "action_items": [
                "Obtain segment disclosure and confirm concentration",
                "Reconcile revenue recognition against collections",
                "Meet the incoming finance leadership",
            ],
        },

        "ConflictResolution": {
            "contradiction_summary": "Reported revenue acceleration against withdrawn guidance.",
            "resolution": (
                "Both are probably true and that is the finding. Revenue recognised "
                "in the quarter can accelerate while management loses confidence in "
                "the next two. The falling free cash flow is the tie-breaker, so the "
                "growth is treated as real but low quality."),
            "confidence": 0.72,
            "action": "keep_both",
        },
    }


class OfflineFixtureError(RuntimeError):
    """A fixture no longer matches its schema."""


def _build(schema: Type[BaseModel], data: dict) -> BaseModel:
    """Validate, and raise on a mismatch.

    An earlier version fell back to `model_construct` here. The pipeline then
    ran to completion while three of four agents degraded to "analysis
    incomplete", and the report still looked plausible. A demo that hides its
    own breakage is worse than no demo.
    """
    try:
        return schema(**data)
    except Exception as e:
        raise OfflineFixtureError(
            f"offline fixture for {schema.__name__} no longer matches the schema: {e}"
        ) from e


class _OfflineStructured:
    """What `.with_structured_output(schema)` returns."""

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def invoke(self, prompt: Any, *_a, **_kw) -> BaseModel:
        company = _company_from(prompt)
        fx = _fixtures(company, _seed(company)).get(self.schema.__name__)
        if fx is None:
            logger.info("no offline fixture for %s, returning an empty instance",
                        self.schema.__name__)
            return self.schema.model_construct()
        return _build(self.schema, fx)

    __call__ = invoke


class OfflineChatModel:
    """Enough of a LangChain chat model for this pipeline, and nothing more."""

    def __init__(self, model_name: str = "offline", temperature: float = 0.0,
                 max_tokens: int = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def with_structured_output(self, schema: Type[BaseModel], **_kw):
        return _OfflineStructured(schema)

    def invoke(self, prompt: Any, *_a, **_kw):
        class _Msg:
            content = ("Offline provider. Set LLM_PROVIDER=google and a "
                       "GOOGLE_API_KEY for real analysis.")
        return _Msg()


def get_offline_llm(model_name: str, temperature: float, max_tokens: int):
    return OfflineChatModel(model_name, temperature, max_tokens)
