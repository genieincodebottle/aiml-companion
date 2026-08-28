"""
Tests for the Quality Gate agent (pure Python, no LLM needed).
Run: pytest tests/test_quality_gate.py -v
"""
import pytest


def test_quality_gate_no_sources():
    from src.agents.quality_gate import quality_gate
    result = quality_gate({"sources": []})
    assert result["quality_score"] == 0.0
    assert result["quality_passed"] is False


def test_quality_gate_good_sources():
    from src.agents.quality_gate import quality_gate
    sources = [
        {"title": "AI Research", "url": "https://arxiv.org/abs/2401.001", "snippet": "This paper presents a comprehensive analysis of large language models and their performance metrics across 50 benchmarks."},
        {"title": "ML Guide", "url": "https://en.wikipedia.org/wiki/Machine_learning", "snippet": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve through experience."},
    ]
    result = quality_gate({"sources": sources})
    assert result["quality_score"] > 0.4
    assert result["quality_passed"] is True
    assert len(result["source_ranking"]) == 2


def test_quality_gate_low_quality_sources():
    from src.agents.quality_gate import quality_gate
    sources = [
        {"title": "", "url": "https://reddit.com/r/test", "snippet": "lol"},
        {"title": "", "url": "https://quora.com/q", "snippet": "idk"},
    ]
    result = quality_gate({"sources": sources})
    assert result["quality_score"] < 0.4
    assert result["quality_passed"] is False


def test_quality_gate_source_ranking_sorted():
    from src.agents.quality_gate import quality_gate
    sources = [
        {"title": "Reddit Post", "url": "https://reddit.com/r/ml", "snippet": "short"},
        {"title": "ArXiv Paper", "url": "https://arxiv.org/abs/123", "snippet": "This comprehensive study of 10000 models shows 95% accuracy improvement in neural network architectures."},
    ]
    result = quality_gate({"sources": sources})
    ranking = result["source_ranking"]
    assert ranking[0]["combined_score"] >= ranking[1]["combined_score"]


def test_domain_score_known():
    from src.agents.quality_gate import _domain_score
    assert _domain_score("https://arxiv.org/abs/123") == 0.95
    assert _domain_score("https://en.wikipedia.org/wiki/Test") == 0.9
    assert _domain_score("https://reddit.com/r/test") == 0.3


def test_domain_score_unknown():
    from src.agents.quality_gate import _domain_score
    assert _domain_score("https://unknownsite.xyz/page") == 0.5


def test_snippet_score_empty():
    from src.agents.quality_gate import _snippet_score
    assert _snippet_score("") == 0.0


def test_snippet_score_with_data():
    from src.agents.quality_gate import _snippet_score
    # Snippet with numbers and tech terms should score higher
    score = _snippet_score("The model achieved 95% accuracy on the training data using a neural network with 1000 parameters")
    assert score > 0.3
