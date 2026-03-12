"""
Web search tools with provider abstraction, caching, and rate limiting.

Supports:
- Tavily (free tier: 1000 searches/month)
- DuckDuckGo (free, no API key)
- Serper (if API key provided)

All providers return a unified SearchResult format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_search_config, load_config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from any provider."""
    title: str
    url: str
    snippet: str
    source: str  # Provider name
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "relevance_score": self.relevance_score,
        }


class SearchCache:
    """SQLite-backed search result cache with TTL."""

    def __init__(self, db_path: str = "data/cache.db", ttl_hours: int = 24, max_entries: int = 1000):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self.max_entries = max_entries

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    results TEXT,
                    created_at REAL,
                    hit_count INTEGER DEFAULT 0
                )
            """)

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> list[dict] | None:
        query_hash = self._hash_query(query)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT results, created_at FROM search_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()

        if row is None:
            return None

        results_json, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            # Expired
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (query_hash,))
            return None

        # Update hit count
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE search_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                (query_hash,),
            )

        return json.loads(results_json)

    def set(self, query: str, results: list[dict]) -> None:
        query_hash = self._hash_query(query)
        self._evict_if_needed()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO search_cache (query_hash, query, results, created_at, hit_count)
                   VALUES (?, ?, ?, ?, 0)""",
                (query_hash, query, json.dumps(results), time.time()),
            )

    def _evict_if_needed(self):
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
            if count >= self.max_entries:
                # LRU eviction: delete oldest 10%
                delete_count = max(1, self.max_entries // 10)
                conn.execute(
                    "DELETE FROM search_cache WHERE query_hash IN "
                    "(SELECT query_hash FROM search_cache ORDER BY created_at ASC LIMIT ?)",
                    (delete_count,),
                )
                logger.info("Cache eviction: removed %d entries", delete_count)

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM search_cache")


class WebSearchTool:
    """Unified web search with provider fallback chain and caching.

    Usage:
        search = WebSearchTool()
        results = search.search("Tesla financial performance 2024")
    """

    def __init__(self, config: dict | None = None):
        self.config = config or get_search_config()
        self.provider = self.config.get("provider", "duckduckgo")
        self.max_results = self.config.get("max_results_per_query", 5)

        # Initialize cache
        cache_config = load_config().get("cache", {})
        self.cache = None
        if cache_config.get("enabled", True):
            try:
                self.cache = SearchCache(
                    db_path=cache_config.get("db_path", "data/cache.db"),
                    ttl_hours=cache_config.get("ttl_hours", 24),
                    max_entries=cache_config.get("max_entries", 1000),
                )
            except Exception as e:
                logger.warning("Cache initialization failed: %s. Continuing without cache.", e)

        # Rate limiting: simple in-memory token bucket
        self._last_search_time = 0.0
        self._min_interval = 1.0  # Minimum seconds between searches

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        """Search the web with caching and provider fallback.

        Args:
            query: Search query string.
            max_results: Override max results per query.

        Returns:
            List of SearchResult objects.
        """
        max_results = max_results or self.max_results

        # Check cache first
        if self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                logger.debug("Cache hit for: %s", query[:50])
                return [SearchResult(**r) for r in cached]

        # Rate limiting
        elapsed = time.time() - self._last_search_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_search_time = time.time()

        # Try providers in order: configured -> duckduckgo fallback
        providers = self._get_provider_chain()
        results = []

        for provider_name, provider_fn in providers:
            try:
                results = provider_fn(query, max_results)
                if results:
                    logger.info("Search via %s: %d results for '%s'", provider_name, len(results), query[:50])
                    break
            except Exception as e:
                logger.warning("Search provider %s failed: %s", provider_name, e)
                continue

        if not results:
            logger.error("All search providers failed for: %s", query[:50])
            return []

        # Cache results
        if self.cache and results:
            self.cache.set(query, [r.to_dict() for r in results])

        return results

    def _get_provider_chain(self) -> list[tuple[str, Any]]:
        """Build ordered provider fallback chain."""
        chain = []

        if self.provider == "tavily" and os.getenv("TAVILY_API_KEY"):
            chain.append(("tavily", self._search_tavily))
        if self.provider == "serper" and os.getenv("SERPER_API_KEY"):
            chain.append(("serper", self._search_serper))

        # DuckDuckGo is always available as fallback (no API key needed)
        chain.append(("duckduckgo", self._search_duckduckgo))

        return chain

    def _search_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using Tavily API."""
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        search_depth = self.config.get("search_depth", "advanced")
        exclude = self.config.get("exclude_domains", [])

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            exclude_domains=exclude,
        )

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:500],
                source="tavily",
                relevance_score=r.get("score", 0.0),
            )
            for r in response.get("results", [])
        ]

    def _search_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", "")[:500],
                source="duckduckgo",
                relevance_score=0.5,  # DDG doesn't provide relevance scores
            )
            for r in raw_results
        ]

    def _search_serper(self, query: str, max_results: int) -> list[SearchResult]:
        """Search using Serper API."""
        import requests

        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", "")[:500],
                source="serper",
                relevance_score=0.5,
            )
            for r in data.get("organic", [])
        ]
