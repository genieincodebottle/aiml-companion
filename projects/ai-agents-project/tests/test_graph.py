"""
Tests for the Graph wiring and routing functions.
Run: pytest tests/test_graph.py -v
"""
import pytest


def test_build_graph_compiles():
    from src.agents.graph import build_graph
    app = build_graph()
    assert app is not None


def test_route_after_quality_pass():
    from src.agents.graph import route_after_quality
    result = route_after_quality({"quality_passed": True})
    assert result == "analyst"


def test_route_after_quality_fail():
    from src.agents.graph import route_after_quality
    result = route_after_quality({
        "quality_passed": False,
        "sub_topics": ["topic1"],
        "search_queries_used": ["topic1"],
    })
    assert result == "retry_researcher"


def test_route_after_quality_already_retried():
    from src.agents.graph import route_after_quality
    result = route_after_quality({
        "quality_passed": False,
        "sub_topics": ["topic1"],
        "search_queries_used": ["topic1", "broad topic1"],
    })
    assert result == "analyst"


def test_route_after_review_pass():
    from src.agents.graph import route_after_review
    from langgraph.graph import END
    result = route_after_review({"review": {"passed": True}})
    assert result == END


def test_route_after_review_fail():
    from src.agents.graph import route_after_review
    result = route_after_review({
        "review": {"passed": False},
        "revision_count": 0,
    })
    assert result == "writer"


def test_route_after_review_max_revisions():
    from src.agents.graph import route_after_review
    from langgraph.graph import END
    result = route_after_review({
        "review": {"passed": False},
        "revision_count": 2,
    })
    assert result == END


def test_route_to_researchers():
    from src.agents.graph import route_to_researchers
    from langgraph.types import Send
    result = route_to_researchers({
        "sub_topics": ["topic1", "topic2", "topic3"],
        "query": "original",
        "token_count": 0,
    })
    assert len(result) == 3
    assert all(isinstance(s, Send) for s in result)


def test_config_loader():
    from src.config import load_config, get_model_name, get_pipeline_config
    config = load_config()
    assert "model" in config
    assert get_model_name() == "gemini-2.5-flash"
    pipeline = get_pipeline_config()
    assert "max_sub_topics" in pipeline
    assert "max_revisions" in pipeline
