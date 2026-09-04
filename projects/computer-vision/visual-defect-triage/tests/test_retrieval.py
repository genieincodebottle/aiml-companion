"""Cosine retrieval, and the stale-index refusal."""
import numpy as np
import pytest

from src.errors import IndexStale
from src.retrieval.index import FlatIndex, build
from src.retrieval.service import RetrievalService


def _fixture():
    rng = np.random.default_rng(4)
    vecs = rng.normal(size=(200, 32)).astype("float32")
    ids = [f"img_{i}" for i in range(200)]
    meta = {i: {"final_ruling": "scratch", "batch_id": "b0"} for i in ids}
    return vecs, ids, meta


def test_a_vector_is_its_own_nearest_neighbour():
    vecs, ids = _fixture()[:2]
    index = FlatIndex(vecs)
    sims, idx = index.search(vecs[7], k=1)
    assert int(idx[0]) == 7
    assert sims[0] == pytest.approx(1.0, abs=1e-4)


def test_similarity_is_bounded_like_a_cosine():
    vecs = _fixture()[0]
    sims, _ = FlatIndex(vecs).search(vecs[0], k=10)
    assert np.all(sims <= 1.0 + 1e-5) and np.all(sims >= -1.0 - 1e-5)


def test_service_refuses_a_stale_index():
    vecs, ids, meta = _fixture()
    index, ids = build(vecs, ids)
    with pytest.raises(IndexStale):
        RetrievalService(index, ids, meta, built_with="an-older-backbone",
                         version="vitb16-augreg2-v3")


def test_service_returns_rulings_not_just_indices():
    vecs, ids, meta = _fixture()
    index, ids = build(vecs, ids)
    svc = RetrievalService(index, ids, meta, built_with="v1", version="v1")
    ns = svc.neighbours(vecs[3], k=5)
    assert len(ns) == 5
    assert ns[0].ruling.value == "scratch"
