"""
Tests for the Research Cache (SQLite).
Run: pytest tests/test_cache.py -v
"""
import pytest
import os
import tempfile


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    """Use a temporary database for each test."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_cache.db")
    monkeypatch.setattr("src.cache.research_cache.DB_PATH", db_path)
    yield
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass


def test_cache_miss():
    from src.cache.research_cache import get_cached_sources
    result = get_cached_sources("nonexistent query")
    assert result is None


def test_cache_write_and_hit():
    from src.cache.research_cache import cache_sources, get_cached_sources
    sources = [
        {"title": "Test", "url": "https://example.com", "snippet": "content"},
    ]
    cache_sources("test query", sources)
    result = get_cached_sources("test query")
    assert result is not None
    assert len(result) == 1
    assert result[0]["title"] == "Test"


def test_cache_case_insensitive():
    from src.cache.research_cache import cache_sources, get_cached_sources
    cache_sources("Test Query", [{"title": "A"}])
    result = get_cached_sources("test query")
    assert result is not None


def test_cache_stats():
    from src.cache.research_cache import cache_sources, get_cache_stats
    cache_sources("q1", [{"title": "A", "url": "https://a.com"}])
    cache_sources("q2", [{"title": "B", "url": "https://b.com"}])
    stats = get_cache_stats()
    assert stats["cached_queries"] == 2
    assert stats["indexed_sources"] == 2


def test_clear_cache():
    from src.cache.research_cache import cache_sources, clear_cache, get_cache_stats
    cache_sources("q1", [{"title": "A"}])
    clear_cache()
    stats = get_cache_stats()
    assert stats["cached_queries"] == 0
