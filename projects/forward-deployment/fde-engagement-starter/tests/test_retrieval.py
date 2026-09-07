"""RUBRIC: hybrid retrieval beats the naive baseline on THEIR corpus.

These fail until you implement src/retrieval/hybrid.py.

The claim "we use hybrid retrieval" is worth nothing in an interview. The claim
"hybrid moved recall at 5 from X to Y on the customer's own policy corpus, here is
the ablation" is the whole difference. These tests make the ablation mandatory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval.hybrid import (
    Document,
    RetrievalConfig,
    evaluate_recall,
    load_corpus,
    search,
)

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "customer" / "data" / "policies"

# Queries whose answer lives in a specific policy document. Keep these honest:
# they should be phrased the way a dispatcher would ask, not the way the document
# is written, or you are only testing string matching.
QUERIES: list[tuple[str, str]] = [
    ("how many times do we retry a failed delivery", "delivery-sla"),
    ("customer wants money back for a late parcel", "refund-policy"),
    ("can we ship lithium batteries", "hazardous-goods"),
    ("driver says the street number is wrong", "address-correction"),
    ("who signs off a goodwill credit", "escalation-matrix"),
]


def _corpus() -> list[Document]:
    return load_corpus(POLICIES)


def test_corpus_loads():
    corpus = _corpus()
    assert len(corpus) >= 5, "five policy documents"
    assert all(isinstance(d, Document) and d.text.strip() for d in corpus)


def test_doc_ids_are_human_readable():
    """You will be reading these out in a demo."""
    for doc in _corpus():
        assert doc.doc_id, "every document needs an id"
        assert not doc.doc_id.isdigit(), (
            "bare numeric ids are unciteable in a demo. Keep the filename."
        )


def test_search_respects_top_k():
    corpus = _corpus()
    hits = search(QUERIES[0][0], corpus, RetrievalConfig(top_k=3))
    assert len(hits) <= 3


def test_search_returns_scores_in_descending_order():
    corpus = _corpus()
    hits = search(QUERIES[0][0], corpus, RetrievalConfig(top_k=5))
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True), "results must be ranked"


def test_disabling_every_stage_is_an_error_not_an_empty_list():
    """A retriever that silently returns nothing is a wasted afternoon."""
    corpus = _corpus()
    with pytest.raises(ValueError):
        search(
            "anything",
            corpus,
            RetrievalConfig(use_lexical=False, use_dense=False),
        )


def test_stages_can_be_ablated_independently():
    """You cannot claim hybrid wins unless you can run each half alone."""
    corpus = _corpus()
    lexical_only = RetrievalConfig(use_lexical=True, use_dense=False)
    dense_only = RetrievalConfig(use_lexical=False, use_dense=True)
    assert search(QUERIES[0][0], corpus, lexical_only)
    assert search(QUERIES[0][0], corpus, dense_only)


def test_hybrid_is_at_least_as_good_as_dense_alone():
    """The measurement that earns the sentence you say in the interview.

    If this fails for you, do not skip it. Either the fusion is wrong or the corpus
    genuinely does not benefit, and finding out which is the exercise.
    """
    corpus = _corpus()
    dense = evaluate_recall(QUERIES, corpus, RetrievalConfig(use_lexical=False))
    hybrid = evaluate_recall(QUERIES, corpus, RetrievalConfig())
    assert hybrid >= dense, (
        f"hybrid recall {hybrid:.2f} is below dense-only {dense:.2f}. "
        "Check your fusion before you claim hybrid in a demo."
    )


def test_retrieval_finds_the_contradiction_source_documents():
    """Two policy documents disagree about redelivery attempts.

    A retriever that surfaces only one of them lets the model answer confidently
    and wrongly. Finding both is what makes the contradiction visible.
    """
    corpus = _corpus()
    hits = search(
        "how many redelivery attempts are allowed", corpus, RetrievalConfig(top_k=5)
    )
    ids = " ".join(doc.doc_id.lower() for doc, _ in hits)
    assert "sla" in ids and "escalation" in ids, (
        "both documents that speak to redelivery attempts should surface. "
        "One of them contradicts the other, and the customer does not know."
    )
