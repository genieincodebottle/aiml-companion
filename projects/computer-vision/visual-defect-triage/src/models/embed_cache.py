"""Write each embedding to disk once, keyed by image and version.

Putting the version in the directory path means a backbone change cannot
silently reuse old vectors. It misses the cache and recomputes.
"""
from pathlib import Path

import numpy as np

from src.models.backbone import EMBEDDING_VERSION

CACHE = Path("artifacts/embeddings")


def path_for(image_id: str, version: str = EMBEDDING_VERSION) -> Path:
    return CACHE / version / f"{image_id}.npy"


def get(image_id: str, version: str = EMBEDDING_VERSION):
    p = path_for(image_id, version)
    return np.load(p) if p.exists() else None


def put(image_id: str, vec, version: str = EMBEDDING_VERSION) -> Path:
    p = path_for(image_id, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(vec, dtype="float32"))
    return p
