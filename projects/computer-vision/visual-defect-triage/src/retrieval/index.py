"""Nearest-neighbour index over normalised embeddings.

Normalise, then use inner product, and you have cosine similarity without a
custom metric. Exhaustive search sounds slow and is not at this scale: 40,000
vectors of 768 dimensions is about 3 ms on a CPU. Reach for an approximate index
when exhaustive search is actually too slow, not before.

faiss is used when installed. The numpy fallback is exact and identical, so the
tests and the demo run anywhere.
"""
import numpy as np


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class FlatIndex:
    """Exact cosine index. Same results with or without faiss."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = _l2_normalise(vectors)
        self._faiss = None
        try:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss = index
        except ImportError:
            pass

    def search(self, vec, k: int = 8):
        q = _l2_normalise(vec)
        if self._faiss is not None:
            sims, idx = self._faiss.search(q, k)
            return sims[0], idx[0]
        sims = (self.vectors @ q[0]).astype("float32")
        idx = np.argsort(-sims)[:k]
        return sims[idx], idx


def build(embeddings, ids: list[str]) -> tuple[FlatIndex, list[str]]:
    return FlatIndex(np.asarray(embeddings)), list(ids)
