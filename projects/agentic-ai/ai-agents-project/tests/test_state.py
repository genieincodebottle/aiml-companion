"""
Tests for the State schema and Pydantic models.
Run: pytest tests/test_state.py -v
"""
import pytest


def test_default_state_has_all_fields():
    from src.models.state import default_state
    state = default_state("test query")
    assert state["query"] == "test query"
    assert state["sources"] == []
    assert state["sub_topics"] == []
    assert state["quality_score"] == 0.0
    assert state["revision_count"] == 0
    assert state["token_count"] == 0
    assert state["pipeline_trace"] == []


def test_planner_output_schema():
    from src.models.state import PlannerOutput
    output = PlannerOutput(
        sub_topics=["topic1", "topic2"],
        research_plan="Research both topics independently",
    )
    assert len(output.sub_topics) == 2
    assert output.research_plan


def test_planner_output_max_topics():
    from src.models.state import PlannerOutput
    output = PlannerOutput(
        sub_topics=["t1", "t2", "t3"],
        research_plan="plan",
    )
    assert len(output.sub_topics) <= 3


def test_review_output_schema():
    from src.models.state import ReviewOutput
    output = ReviewOutput(
        score=8,
        issues=["Missing conclusion"],
        suggestions=["Add summary"],
        passed=True,
    )
    assert output.score == 8
    assert output.passed is True
    assert len(output.issues) == 1


def test_review_output_score_bounds():
    from src.models.state import ReviewOutput
    with pytest.raises(Exception):
        ReviewOutput(score=0, issues=[], suggestions=[], passed=False)
    with pytest.raises(Exception):
        ReviewOutput(score=11, issues=[], suggestions=[], passed=False)


def test_claim_output_schema():
    from src.models.state import ClaimOutput
    claim = ClaimOutput(
        claim="AI is advancing rapidly",
        source_idx=1,
        confidence="high",
        evidence="Multiple studies show...",
    )
    assert claim.source_idx == 1
    assert claim.confidence == "high"
