"""
ChromaDB vector store - three collections for three memory types.

Collections:
  1. claims_long_term   - all completed claims (long-term memory)
     doc = claim description, metadata = outcome, amount, fraud_score, etc.
     Use: "find similar past claims to inform the current decision"

  2. claims_episodic    - notable episodes (human overrides, fraud confirmed, appeals)
     doc = narrative of what happened, metadata = claim_id, event_type
     Use: "last time a similar claim was overridden by a reviewer, why?"

  3. fraud_patterns     - known fraud narratives + red flags
     doc = fraud pattern description, metadata = risk_level, frequency
     Use: "does this claim's story match known fraud patterns?"

Persistence: data/memory/chroma/ (survives restarts, gitignored)

Learner takeaway:
  Long-term memory = "what the system has seen before"
  Episodic memory = "specific events worth remembering"
  Short-term memory = LangGraph's ClaimsState (already exists - the state
  dict IS the working memory for the current pipeline run)
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from src.memory.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory" / "chroma"
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(_PERSIST_DIR))
        logger.info("ChromaDB initialized at %s", _PERSIST_DIR)
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a named collection with HuggingFace embeddings."""
    ef = get_embedding_function()
    kwargs = {"name": name}
    if ef is not None:
        kwargs["embedding_function"] = ef
    return _get_client().get_or_create_collection(**kwargs)


# ── Collection accessors ────────────────────────────────────────────────────

def long_term() -> chromadb.Collection:
    """All completed claims - used for similar-claim retrieval."""
    return get_collection("claims_long_term")


def episodic() -> chromadb.Collection:
    """Notable episodes - human overrides, confirmed fraud, appeals."""
    return get_collection("claims_episodic")


def fraud_knowledge() -> chromadb.Collection:
    """Known fraud patterns and red flags."""
    return get_collection("fraud_patterns")
