"""
Embedding model - HuggingFace sentence-transformers.

Uses `all-MiniLM-L6-v2` (80 MB, ~14K tokens/sec on CPU).
Downloaded once on first use, cached at ~/.cache/huggingface/.

Learner takeaway:
  Embeddings convert text into dense vectors so we can compute
  semantic similarity (cosine distance) between claims. Two claims
  about "rear-ended at a stop sign" and "hit from behind at an
  intersection" have different words but similar embeddings.
"""
from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality


@lru_cache(maxsize=1)
def get_embedding_function():
    """Return a ChromaDB-compatible embedding function using HuggingFace.

    ChromaDB accepts any object with an __call__(texts: list[str]) -> list[list[float]]
    interface. SentenceTransformer fits this via .encode().
    """
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        ef = SentenceTransformerEmbeddingFunction(model_name=DEFAULT_MODEL)
        logger.info("Loaded HuggingFace embedding model: %s", DEFAULT_MODEL)
        return ef
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Using ChromaDB default embeddings. "
            "Run: uv pip install sentence-transformers"
        )
        return None  # ChromaDB will use its built-in default
