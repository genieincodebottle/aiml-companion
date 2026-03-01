"""
End-to-end pipeline tests with mocked LLM calls.

These tests verify the full graph routing, state accumulation,
parallel fan-out, quality gate retry, and reviewer refinement loop
without making real API calls.

Run: pytest tests/test_e2e_pipeline.py -v
"""
import pytest
from unittest.mock import patch, MagicMock

from src.models.state import default_state, ResearchState
from src.agents.graph import (
    build_graph, route_to_researchers, route_after_quality, route_after_review,
)
from src.agents.quality_gate import quality_gate, _domain_score, _snippet_score
from src.guardrails import scrub_pii, check_budget, TOKEN_BUDGET


# === Fixtures ===

MOCK_SOURCES_GOOD = [
    {"title": "AI Agents Survey", "url": "https://arxiv.org/abs/2401.0001",
     "snippet": "A comprehensive study of AI agent architectures and their performance in 2024 research benchmarks.", "tool": "tavily"},
    {"title": "LangGraph Overview", "url": "https://github.com/langchain-ai/langgraph",
     "snippet": "LangGraph is a framework for building stateful multi-agent applications with explicit data flow.", "tool": "tavily"},
    {"title": "Agent Design Patterns", "url": "https://en.wikipedia.org/wiki/Intelligent_agent",
     "snippet": "An intelligent agent observes its environment through sensors and acts upon it through actuators to achieve goals.", "tool": "wikipedia"},
]

MOCK_SOURCES_BAD = [
    {"title": "Random post", "url": "https://reddit.com/r/ai/123",
     "snippet": "lol", "tool": "tavily"},
    {"title": "Short", "url": "https://quora.com/q/456",
     "snippet": "idk", "tool": "tavily"},
]

MOCK_PLANNER_RESPONSE = MagicMock(
    sub_topics=["AI agent architectures", "LangGraph patterns"],
    research_plan="Research agent architectures and LangGraph patterns separately.",
)

MOCK_ANALYST_RESPONSE = MagicMock(
    claims=[
        MagicMock(claim="AI agents use LLM-based reasoning", source_idx=1, confidence="high", evidence="Source 1 states..."),
        MagicMock(claim="LangGraph enables stateful agents", source_idx=2, confidence="high", evidence="Source 2 describes..."),
    ],
    conflicts=[],
)

MOCK_SYNTHESIS_RESPONSE = MagicMock(
    content="AI agents represent a paradigm shift in software design. LangGraph enables stateful multi-agent pipelines.",
    response_metadata={"token_metadata": {"output_token_count": {"total_tokens": 150}}},
)

MOCK_WRITER_RESPONSE = MagicMock(
    content="## Introduction\nAI agents are transforming software.\n## Key Findings\n...\n## Sources\n[1] arxiv.org",
    response_metadata={"token_metadata": {"output_token_count": {"total_tokens": 300}}},
)

MOCK_REVIEW_PASS = MagicMock(score=8, issues=[], suggestions=[], passed=True)
MOCK_REVIEW_FAIL = MagicMock(score=5, issues=["Missing depth"], suggestions=["Add more examples"], passed=False)


# === E2E: Full Pipeline with Mocked LLM ===

class TestFullPipelineMocked:
    """Test the full 8-agent pipeline end-to-end with mocked LLM calls."""

    def _make_mock_llm(self, structured_response):
        """Create a mock ChatGoogleGenerativeAI that returns structured output."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = structured_response
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm.invoke.return_value = structured_response
        return mock_llm

    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.wiki_search")
    @patch("src.agents.researcher.select_tools")
    @patch("src.agents.planner.ChatGoogleGenerativeAI")
    @patch("src.agents.analyst.ChatGoogleGenerativeAI")
    @patch("src.agents.synthesizer.ChatGoogleGenerativeAI")
    @patch("src.agents.writer.ChatGoogleGenerativeAI")
    @patch("src.agents.reviewer.ChatGoogleGenerativeAI")
    def test_full_pipeline_happy_path(
        self, mock_reviewer_llm, mock_writer_llm, mock_synth_llm,
        mock_analyst_llm, mock_planner_llm,
        mock_select_tools, mock_wiki, mock_web
    ):
        """Full pipeline: planner -> researchers -> quality gate -> analyst -> synth -> writer -> reviewer (pass)."""
        # Setup mocks
        mock_planner_llm.return_value = self._make_mock_llm(MOCK_PLANNER_RESPONSE)
        mock_analyst_llm.return_value = self._make_mock_llm(MOCK_ANALYST_RESPONSE)
        mock_synth_llm.return_value = self._make_mock_llm(MOCK_SYNTHESIS_RESPONSE)
        mock_writer_llm.return_value = self._make_mock_llm(MOCK_WRITER_RESPONSE)
        mock_reviewer_llm.return_value = self._make_mock_llm(MOCK_REVIEW_PASS)

        mock_select_tools.return_value = ["tavily", "wikipedia"]
        mock_web.return_value = MOCK_SOURCES_GOOD[:2]
        mock_wiki.return_value = [MOCK_SOURCES_GOOD[2]]

        # Run pipeline
        app = build_graph()
        initial = default_state("What are AI agent architectures?")
        result = app.invoke(initial)

        # Verify state populated
        assert len(result["sub_topics"]) >= 1, "Planner should produce sub-topics"
        assert len(result["sources"]) >= 1, "Researchers should find sources"
        assert result["quality_score"] > 0, "Quality gate should score sources"
        assert result["quality_passed"], "Good sources should pass quality gate"
        assert len(result["key_claims"]) >= 1, "Analyst should extract claims"
        assert result["synthesis"], "Synthesizer should produce synthesis"
        assert result["current_draft"], "Writer should produce a draft"
        assert result["review"], "Reviewer should produce a review"
        assert result["final_report"], "Passing review should produce final report"
        assert result["token_count"] > 0, "Token count should accumulate"
        assert len(result["pipeline_trace"]) >= 5, "Trace should have entries for each agent"

    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.wiki_search")
    @patch("src.agents.researcher.select_tools")
    @patch("src.agents.planner.ChatGoogleGenerativeAI")
    @patch("src.agents.analyst.ChatGoogleGenerativeAI")
    @patch("src.agents.synthesizer.ChatGoogleGenerativeAI")
    @patch("src.agents.writer.ChatGoogleGenerativeAI")
    @patch("src.agents.reviewer.ChatGoogleGenerativeAI")
    def test_pipeline_trace_agent_order(
        self, mock_reviewer_llm, mock_writer_llm, mock_synth_llm,
        mock_analyst_llm, mock_planner_llm,
        mock_select_tools, mock_wiki, mock_web
    ):
        """Pipeline trace should contain agents in correct execution order."""
        mock_planner_llm.return_value = self._make_mock_llm(MOCK_PLANNER_RESPONSE)
        mock_analyst_llm.return_value = self._make_mock_llm(MOCK_ANALYST_RESPONSE)
        mock_synth_llm.return_value = self._make_mock_llm(MOCK_SYNTHESIS_RESPONSE)
        mock_writer_llm.return_value = self._make_mock_llm(MOCK_WRITER_RESPONSE)
        mock_reviewer_llm.return_value = self._make_mock_llm(MOCK_REVIEW_PASS)

        mock_select_tools.return_value = ["tavily"]
        mock_web.return_value = MOCK_SOURCES_GOOD[:2]
        mock_wiki.return_value = []

        app = build_graph()
        result = app.invoke(default_state("Test query"))

        trace_agents = [t["agent"] for t in result["pipeline_trace"]]

        # Planner must be first
        assert trace_agents[0] == "planner"

        # Researchers before quality_gate
        qg_idx = trace_agents.index("quality_gate")
        researcher_indices = [i for i, a in enumerate(trace_agents) if a == "researcher"]
        assert all(ri < qg_idx for ri in researcher_indices), "Researchers must run before quality gate"

        # Analyst after quality gate
        analyst_idx = trace_agents.index("analyst")
        assert analyst_idx > qg_idx, "Analyst must run after quality gate"

        # Writer before reviewer
        writer_idx = trace_agents.index("writer")
        reviewer_idx = trace_agents.index("reviewer")
        assert writer_idx < reviewer_idx, "Writer must run before reviewer"


# === E2E: Quality Gate Retry Flow ===

class TestQualityGateRetryFlow:
    """Test that bad sources trigger retry routing."""

    def test_bad_sources_fail_quality_gate(self):
        """Low-quality sources (reddit, quora, short snippets) should fail quality gate."""
        state = {"sources": MOCK_SOURCES_BAD}
        result = quality_gate(state)
        assert not result["quality_passed"], "Bad sources should not pass quality gate"
        assert result["quality_score"] < 0.4, "Quality score should be below threshold"

    def test_good_sources_pass_quality_gate(self):
        """High-quality sources (arxiv, wikipedia) should pass quality gate."""
        state = {"sources": MOCK_SOURCES_GOOD}
        result = quality_gate(state)
        assert result["quality_passed"], "Good sources should pass quality gate"
        assert result["quality_score"] >= 0.4, "Quality score should be above threshold"

    def test_quality_gate_routing_retry(self):
        """Route after quality: low score + first attempt -> retry_researcher."""
        state = {
            "quality_passed": False,
            "sub_topics": ["topic1", "topic2"],
            "search_queries_used": ["topic1", "topic2"],  # Same count = not retried yet
        }
        assert route_after_quality(state) == "retry_researcher"

    def test_quality_gate_routing_already_retried(self):
        """Route after quality: low score + already retried -> analyst (prevent infinite loop)."""
        state = {
            "quality_passed": False,
            "sub_topics": ["topic1"],
            "search_queries_used": ["topic1", "topic1 comprehensive overview analysis"],
        }
        assert route_after_quality(state) == "analyst"

    def test_source_ranking_order(self):
        """Source ranking should be sorted by combined score descending."""
        state = {"sources": MOCK_SOURCES_GOOD + MOCK_SOURCES_BAD}
        result = quality_gate(state)
        ranking = result["source_ranking"]
        scores = [r["combined_score"] for r in ranking]
        assert scores == sorted(scores, reverse=True), "Rankings should be sorted descending"


# === E2E: Reviewer Refinement Loop ===

class TestReviewerRefinementLoop:
    """Test the writer-reviewer loop mechanics."""

    def test_review_pass_ends_pipeline(self):
        """Passing review should route to END."""
        from langgraph.graph import END
        state = {"review": {"passed": True}}
        assert route_after_review(state) == END

    def test_review_fail_loops_to_writer(self):
        """Failing review with revisions remaining should loop to writer."""
        state = {"review": {"passed": False}, "revision_count": 0}
        assert route_after_review(state) == "writer"

    def test_review_fail_max_revisions_ends(self):
        """Failing review at max revisions should route to END (prevent infinite loop)."""
        from langgraph.graph import END
        state = {"review": {"passed": False}, "revision_count": 2}
        assert route_after_review(state) == END

    def test_revision_count_below_max(self):
        """Revision count 1 with max 2 should still loop."""
        state = {"review": {"passed": False}, "revision_count": 1}
        assert route_after_review(state) == "writer"


# === E2E: Parallel Fan-Out ===

class TestParallelFanOut:
    """Test Send() fan-out mechanics."""

    def test_send_creates_one_per_subtopic(self):
        """Route to researchers should create one Send per sub-topic."""
        from langgraph.types import Send
        state = {
            "sub_topics": ["topic A", "topic B", "topic C"],
            "query": "main query",
            "token_count": 0,
        }
        sends = route_to_researchers(state)
        assert len(sends) == 3, "Should create 3 Send objects for 3 sub-topics"
        assert all(isinstance(s, Send) for s in sends)

    def test_send_passes_subtopic_as_query(self):
        """Each Send should pass its sub-topic as the query."""
        state = {
            "sub_topics": ["AI agents", "LangGraph"],
            "query": "main query",
            "token_count": 100,
        }
        sends = route_to_researchers(state)
        args = [s.arg for s in sends]
        assert args[0]["query"] == "AI agents"
        assert args[1]["query"] == "LangGraph"

    def test_send_passes_current_token_count(self):
        """Each Send should pass the current token count for budget checking."""
        state = {
            "sub_topics": ["topic"],
            "query": "main",
            "token_count": 5000,
        }
        sends = route_to_researchers(state)
        assert sends[0].arg["token_count"] == 5000

    def test_fallback_to_original_query(self):
        """If no sub_topics, fall back to the original query."""
        state = {"query": "fallback query", "token_count": 0}
        sends = route_to_researchers(state)
        assert len(sends) == 1
        assert sends[0].arg["query"] == "fallback query"


# === E2E: Budget Enforcement ===

class TestBudgetEnforcement:
    """Test that budget checks prevent over-spending."""

    def test_budget_ok_under_limit(self):
        assert check_budget(1000) is True

    def test_budget_exceeded(self):
        assert check_budget(50000) is False
        assert check_budget(60000) is False

    def test_budget_at_limit(self):
        assert check_budget(49999) is True
        assert check_budget(50000) is False

    def test_custom_budget(self):
        assert check_budget(5000, budget=10000) is True
        assert check_budget(10000, budget=10000) is False


# === E2E: PII Scrubbing ===

class TestPIIScrubbing:
    """Test PII detection and scrubbing in pipeline context."""

    def test_email_scrubbed(self):
        text = "Contact john@example.com for details."
        cleaned, types = scrub_pii(text)
        assert "john@example.com" not in cleaned
        assert "[REDACTED_EMAIL]" in cleaned
        assert "email" in types

    def test_phone_scrubbed(self):
        text = "Call 555-123-4567 now."
        cleaned, types = scrub_pii(text)
        assert "555-123-4567" not in cleaned
        assert "phone" in types

    def test_ssn_scrubbed(self):
        text = "SSN: 123-45-6789"
        cleaned, types = scrub_pii(text)
        assert "123-45-6789" not in cleaned
        assert "ssn" in types

    def test_clean_text_unchanged(self):
        text = "AI agents use LLM-based reasoning for complex tasks."
        cleaned, types = scrub_pii(text)
        assert cleaned == text
        assert types == []

    def test_multiple_pii_types(self):
        text = "john@test.com called 555-111-2222 about SSN 111-22-3333"
        cleaned, types = scrub_pii(text)
        assert len(types) == 3
        assert "john@test.com" not in cleaned
        assert "555-111-2222" not in cleaned
        assert "111-22-3333" not in cleaned


# === E2E: Cache Integration ===

class TestCacheIntegration:
    """Test SQLite cache in pipeline context."""

    def test_cache_roundtrip(self):
        from src.cache.research_cache import cache_sources, get_cached_sources, clear_cache
        clear_cache()

        query = "test cache roundtrip e2e"
        sources = [{"title": "Test", "url": "https://example.com/1", "snippet": "Test snippet"}]

        # Miss on first call
        assert get_cached_sources(query) is None

        # Store
        cache_sources(query, sources)

        # Hit on second call
        cached = get_cached_sources(query)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["title"] == "Test"

        clear_cache()

    def test_cache_stats_update(self):
        from src.cache.research_cache import cache_sources, get_cached_sources, get_cache_stats, clear_cache
        clear_cache()

        stats = get_cache_stats()
        assert stats["cached_queries"] == 0

        cache_sources("stats test query", [{"title": "X", "url": "https://x.com", "snippet": "x"}])
        stats = get_cache_stats()
        assert stats["cached_queries"] == 1
        assert stats["indexed_sources"] == 1

        # Trigger a hit
        get_cached_sources("stats test query")
        stats = get_cache_stats()
        assert stats["total_hits"] == 1

        clear_cache()


# === E2E: State Schema Validation ===

class TestStateSchemaE2E:
    """Test state schema completeness for the full pipeline."""

    def test_default_state_has_all_required_fields(self):
        """Default state must have every field needed by every agent."""
        state = default_state("test")
        required = [
            "query", "sub_topics", "research_plan", "sources",
            "search_queries_used", "quality_score", "quality_passed",
            "key_claims", "conflicts", "synthesis", "source_ranking",
            "drafts", "current_draft", "review", "revision_count",
            "token_count", "errors", "final_report", "pipeline_trace",
        ]
        for field in required:
            assert field in state, f"Missing field: {field}"

    def test_default_state_types(self):
        """Default state values should have correct types."""
        state = default_state("test query")
        assert isinstance(state["query"], str)
        assert isinstance(state["sub_topics"], list)
        assert isinstance(state["sources"], list)
        assert isinstance(state["quality_score"], float)
        assert isinstance(state["quality_passed"], bool)
        assert isinstance(state["review"], dict)
        assert isinstance(state["revision_count"], int)
        assert isinstance(state["token_count"], int)
        assert isinstance(state["errors"], list)
        assert isinstance(state["pipeline_trace"], list)

    def test_graph_compiles_with_state_schema(self):
        """Graph should compile with the ResearchState schema."""
        app = build_graph()
        assert app is not None


# === E2E: Domain and Snippet Scoring ===

class TestScoringHeuristics:
    """Test the pure Python scoring heuristics used by quality gate."""

    def test_high_trust_domains(self):
        assert _domain_score("https://arxiv.org/abs/2401.0001") == 0.95
        assert _domain_score("https://en.wikipedia.org/wiki/AI") == 0.9
        assert _domain_score("https://nature.com/articles/123") == 0.95

    def test_low_trust_domains(self):
        assert _domain_score("https://reddit.com/r/ai") == 0.3
        assert _domain_score("https://quora.com/q/123") == 0.3

    def test_unknown_domain(self):
        assert _domain_score("https://random-blog.com/post") == 0.5

    def test_snippet_quality_long_with_data(self):
        snippet = "The model achieved 95.3% accuracy on the benchmark with training data of 10000 samples."
        score = _snippet_score(snippet)
        assert score > 0.3, "Long snippet with data should score high"

    def test_snippet_quality_empty(self):
        assert _snippet_score("") == 0.0
        assert _snippet_score(None) == 0.0

    def test_snippet_quality_very_short(self):
        score = _snippet_score("lol")
        assert score < 0.1, "Very short snippet should score very low"


# === E2E: Rate Limiter ===

class TestRateLimiterE2E:
    """Test rate limiter behavior."""

    def test_rate_limiter_tracks_calls(self):
        from src.guardrails import RateLimiter
        rl = RateLimiter(max_rpm=600)  # Fast for testing (0.1s interval)
        rl.wait_if_needed()
        rl.wait_if_needed()
        assert rl.total_calls == 2

    def test_rate_limiter_reset(self):
        from src.guardrails import RateLimiter
        rl = RateLimiter(max_rpm=600)
        rl.wait_if_needed()
        assert rl.total_calls == 1
        rl.reset()
        assert rl.total_calls == 0
