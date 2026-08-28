"""Tests for search tools."""

import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.search import SearchResult, SearchCache
import tempfile
import os


class TestSearchResult:
    def test_to_dict(self):
        sr = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="A test result",
            source="test",
            relevance_score=0.9,
        )
        d = sr.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "https://example.com"
        assert d["relevance_score"] == 0.9


class TestSearchCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_cache.db")
        self.cache = SearchCache(db_path=self.db_path, ttl_hours=1, max_entries=10)

    def test_get_miss(self):
        result = self.cache.get("nonexistent query")
        assert result is None

    def test_set_and_get(self):
        data = [{"title": "Test", "url": "https://example.com", "snippet": "test", "source": "test", "relevance_score": 0.5}]
        self.cache.set("test query", data)
        result = self.cache.get("test query")
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    def test_case_insensitive(self):
        data = [{"title": "Test", "url": "https://example.com", "snippet": "test", "source": "test", "relevance_score": 0.5}]
        self.cache.set("Tesla Revenue", data)
        result = self.cache.get("tesla revenue")
        assert result is not None

    def test_clear(self):
        data = [{"title": "Test", "url": "https://example.com", "snippet": "test", "source": "test", "relevance_score": 0.5}]
        self.cache.set("test", data)
        self.cache.clear()
        result = self.cache.get("test")
        assert result is None

    def test_eviction(self):
        cache = SearchCache(db_path=self.db_path, ttl_hours=1, max_entries=5)
        for i in range(6):
            cache.set(f"query_{i}", [{"title": f"Result {i}", "url": "", "snippet": "", "source": "test", "relevance_score": 0.5}])
        # After eviction, most recent should still be there
        result = cache.get("query_5")
        assert result is not None
