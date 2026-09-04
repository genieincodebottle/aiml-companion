"""Load the model, index, and temperature once at startup.

lru_cache on a zero-argument function is the simplest correct singleton in
Python, and it runs the stale-index check exactly once. If the index came from an
older backbone, RetrievalService raises during the first call and the process
fails to serve, which is the right moment to find out.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.models.head import NumpyHead
from src.retrieval.index import FlatIndex
from src.retrieval.service import RetrievalService

RUN_DIR = Path("artifacts/run")


@lru_cache(maxsize=1)
def get_pipeline():
    head = NumpyHead.load(RUN_DIR / "head.npz")
    manifest = json.loads((RUN_DIR / "manifest.json").read_text(encoding="utf-8"))

    vectors = np.load(RUN_DIR / "index_vectors.npy")
    ids = np.load(RUN_DIR / "index_ids.npy", allow_pickle=True).tolist()
    meta = json.loads((RUN_DIR / "index_meta.json").read_text(encoding="utf-8"))

    retrieval = RetrievalService(FlatIndex(vectors), ids, meta,
                                 built_with=manifest["embedding_version"])
    return head, retrieval, manifest["temperature"]
