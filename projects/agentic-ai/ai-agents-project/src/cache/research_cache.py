# ============================================
# Research Cache - SQLite Source & Query Cache
# ============================================
# Caches research results in SQLite to avoid duplicate API calls.
# Different from Content Moderation's ChromaDB (vector similarity for decisions).
# This is a deterministic cache (exact query match, source dedup).
# Teaches: caching for API cost control, SQLite in production pipelines.

import os
import json
import time
import sqlite3
import logging
import hashlib

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "research_cache.db")
CACHE_TTL = 3600 * 24  # 24 hours


def _get_conn() -> sqlite3.Connection:
    """Get or create SQLite connection, ensuring tables exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query TEXT,
            sources_json TEXT,
            created_at REAL,
            hit_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_index (
            url TEXT PRIMARY KEY,
            title TEXT,
            snippet TEXT,
            first_seen REAL,
            last_seen REAL,
            use_count INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    return conn


def _hash_query(query: str) -> str:
    """Create a stable hash for a query string."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def get_cached_sources(query: str) -> list[dict] | None:
    """Check if we have cached results for this query.

    Returns list of source dicts if cache hit (and not expired), None otherwise.
    """
    try:
        conn = _get_conn()
        qhash = _hash_query(query)
        row = conn.execute(
            "SELECT sources_json, created_at FROM query_cache WHERE query_hash = ?",
            (qhash,),
        ).fetchone()

        if row is None:
            return None

        sources_json, created_at = row
        if time.time() - created_at > CACHE_TTL:
            logger.info(f"Cache expired for query: {query[:50]}")
            conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (qhash,))
            conn.commit()
            conn.close()
            return None

        # Update hit count
        conn.execute(
            "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
            (qhash,),
        )
        conn.commit()
        conn.close()

        sources = json.loads(sources_json)
        logger.info(f"Cache HIT for '{query[:50]}': {len(sources)} sources")
        return sources
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None


def cache_sources(query: str, sources: list[dict]) -> None:
    """Store research results in the cache."""
    try:
        conn = _get_conn()
        qhash = _hash_query(query)
        now = time.time()

        # Store query results
        conn.execute(
            "INSERT OR REPLACE INTO query_cache (query_hash, query, sources_json, created_at, hit_count) "
            "VALUES (?, ?, ?, ?, 0)",
            (qhash, query.strip(), json.dumps(sources), now),
        )

        # Update source index
        for src in sources:
            url = src.get("url", "")
            if not url:
                continue
            conn.execute(
                "INSERT INTO source_index (url, title, snippet, first_seen, last_seen, use_count) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(url) DO UPDATE SET last_seen=?, use_count=use_count+1",
                (url, src.get("title", ""), src.get("snippet", "")[:200], now, now, now),
            )

        conn.commit()
        conn.close()
        logger.info(f"Cached {len(sources)} sources for '{query[:50]}'")
    except Exception as e:
        logger.error(f"Cache write error: {e}")


def get_cache_stats() -> dict:
    """Return cache statistics for the UI."""
    try:
        conn = _get_conn()
        query_count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        source_count = conn.execute("SELECT COUNT(*) FROM source_index").fetchone()[0]
        total_hits = conn.execute("SELECT SUM(hit_count) FROM query_cache").fetchone()[0] or 0
        conn.close()
        return {
            "cached_queries": query_count,
            "indexed_sources": source_count,
            "total_hits": total_hits,
        }
    except Exception:
        return {"cached_queries": 0, "indexed_sources": 0, "total_hits": 0}


def clear_cache() -> None:
    """Clear the entire cache (for testing or manual reset)."""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM query_cache")
        conn.execute("DELETE FROM source_index")
        conn.commit()
        conn.close()
        logger.info("Cache cleared")
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
