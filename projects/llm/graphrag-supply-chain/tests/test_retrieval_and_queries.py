"""Tests for the pure logic in retrieval and the query layer.

Nothing here touches Neo4j or a model. What is tested is the code that decides
what gets sent to them: fusion, query sanitising, depth validation, and the
guarantee that no model-supplied string can reach a Cypher label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.graph import queries
from src.retrieval.base import Evidence, RetrievalResult, reciprocal_rank_fusion
from src.retrieval.strategies import STRATEGIES, _lucene

ROOT = Path(__file__).resolve().parent.parent


def _ev(chunk_id: str, kind: str = "text") -> Evidence:
    return Evidence(kind=kind, text=f"text of {chunk_id}", source_id=chunk_id,
                    doc_id=chunk_id.split("::")[0], title="T")


class TestLuceneSanitising:
    def test_escapes_reserved_characters(self):
        # Unescaped, "NW-500" is a Lucene syntax error and the driver surfaces
        # it as an opaque server exception.
        assert "\\-" in _lucene("NW-500")

    def test_preserves_identifiers_as_searchable_tokens(self):
        # Exact identifier match is the entire reason the keyword arm exists.
        assert "500" in _lucene("Which products use NW-500?")

    def test_drops_stopwords(self):
        query = _lucene("What is the status of the finding?")
        assert " OR " in query
        for stop in ("what", "the", "of", "is"):
            assert stop not in query.lower().split(" OR ")

    def test_never_returns_empty(self):
        # An empty Lucene query throws server-side. A query of only stopwords
        # must still produce something runnable.
        assert _lucene("what is the of and").strip()


class TestReciprocalRankFusion:
    def test_agreement_between_lists_outranks_a_single_top_hit(self):
        """The property that makes RRF worth using.

        A chunk both retrievers found in second place should beat one only the
        vector arm found in first, because agreement between two independent
        rankings is stronger evidence than one ranking's confidence.
        """
        fused = reciprocal_rank_fusion(
            {
                "vector": [_ev("a::1"), _ev("shared::1")],
                "keyword": [_ev("b::1"), _ev("shared::1")],
            },
            limit=3,
        )
        assert fused[0].source_id == "shared::1"

    def test_records_which_arms_found_each_item(self):
        fused = reciprocal_rank_fusion(
            {"vector": [_ev("x::1")], "keyword": [_ev("x::1")]}, limit=1
        )
        assert fused[0].retrieved_by == "keyword+vector"

    def test_respects_the_limit(self):
        lists = {"vector": [_ev(f"c::{i}") for i in range(20)]}
        assert len(reciprocal_rank_fusion(lists, limit=5)) == 5

    def test_deduplicates_across_lists(self):
        fused = reciprocal_rank_fusion(
            {"vector": [_ev("x::1")], "keyword": [_ev("x::1")]}, limit=10
        )
        assert len(fused) == 1


class TestContextAssembly:
    def test_graph_facts_are_placed_first(self):
        """Position matters. Burying a three-line derived fact under 8,000
        characters of prose is a measurable way to lose the answer you ran the
        traversal to get."""
        result = RetrievalResult(strategy="hybrid")
        result.evidence = [_ev("doc::1"), _ev("exposure::kaohsiung", kind="graph_fact")]
        context = result.context(max_chars=10000)
        assert context.index("DERIVED FROM THE KNOWLEDGE GRAPH") < context.index("doc::1")

    def test_context_respects_the_character_cap(self):
        result = RetrievalResult(strategy="vector")
        result.evidence = [
            Evidence(kind="text", text="x" * 500, source_id=f"c::{i}", doc_id="D")
            for i in range(20)
        ]
        assert len(result.context(max_chars=1200)) <= 1400  # cap plus separators

    def test_empty_evidence_yields_empty_context(self):
        # The answer layer relies on this to refuse without a model call.
        assert RetrievalResult(strategy="graph").context(max_chars=100) == ""


class TestQueryGuards:
    def test_depth_is_validated(self):
        with pytest.raises(ValueError):
            queries.neighbourhood(0)
        with pytest.raises(ValueError):
            queries.neighbourhood(99)

    def test_depth_is_not_injectable(self):
        """Depth is the one value formatted into Cypher rather than passed as a
        parameter, because Cypher forbids *1..$n. So it is the one value that
        must be proven safe first."""
        with pytest.raises((ValueError, TypeError)):
            queries.neighbourhood("3; MATCH (n) DETACH DELETE n //")

    def test_traversal_never_crosses_into_the_text_subgraph(self):
        """MENTIONS and PART_OF must not appear in any traversal pattern.

        Allowing them means two suppliers become "related" because one PDF
        mentioned both. That is a coincidence, not a relationship, and it is how
        a GraphRAG system starts producing confident nonsense.
        """
        assert "MENTIONS" not in queries.KNOWLEDGE_RELS
        assert "PART_OF" not in queries.KNOWLEDGE_RELS
        pattern = queries.neighbourhood(2)
        assert "MENTIONS" not in pattern and "PART_OF" not in pattern

    def test_every_traversal_query_is_bounded(self):
        """An unbounded expansion through a hub node is a denial of service
        against your own database."""
        for name in ("neighbourhood",):
            assert "LIMIT" in getattr(queries, name)(2)
        for name in ("PRODUCTS_EXPOSED_TO_LOCATION", "SUPPLIER_DOWNSTREAM_IMPACT",
                     "CHUNKS_FOR_ENTITIES", "SUPPLIER_CRITICALITY"):
            assert "LIMIT" in getattr(queries, name)

    def test_values_are_parameterised_not_interpolated(self):
        # No Cypher constant may contain a Python format placeholder for a
        # value; only labels and validated depths are ever formatted in.
        for name in ("VECTOR_SEARCH", "FULLTEXT_CHUNK_SEARCH", "ENTITY_LINK",
                     "CHUNK_ENTITIES", "PRODUCTS_EXPOSED_TO_LOCATION"):
            cypher = getattr(queries, name)
            assert "$" in cypher, f"{name} takes no parameters - is it hardcoding a value?"

    def test_labels_come_from_a_closed_vocabulary(self):
        """Cypher cannot parameterise a label, so labels are formatted in. The
        defence is that callers may only pass a configured type - never a string
        that came from a model."""
        cypher = queries.upsert_entities("Supplier")
        assert "SET e:Supplier" in cypher


class TestGoldenQuestionSet:
    """The benchmark is part of the codebase and is tested like code.

    A benchmark that only contains questions your system wins is marketing.
    These tests assert the set stays honest.
    """

    def setup_method(self):
        with open(ROOT / "data" / "golden_questions.json", encoding="utf-8") as fh:
            self.questions = json.load(fh)["questions"]

    def test_ids_are_unique(self):
        ids = [q["id"] for q in self.questions]
        assert len(ids) == len(set(ids))

    def test_every_question_has_a_declared_expected_winner(self):
        for q in self.questions:
            assert q["expected_advantage"] in {"graph", "vector", "keyword", "hybrid", "tie"}

    def test_the_set_contains_questions_graphrag_should_not_win(self):
        """Without these the evaluation cannot detect a regression in the
        baseline, and every reported advantage is unfalsifiable."""
        losers = [q for q in self.questions
                  if q["expected_advantage"] in {"vector", "keyword", "tie"}]
        assert len(losers) >= 4

    def test_the_set_contains_unanswerable_questions(self):
        assert sum(1 for q in self.questions if q.get("must_refuse")) >= 2

    def test_multi_hop_questions_declare_more_than_one_hop(self):
        for q in self.questions:
            if q["category"] == "multi_hop":
                assert q["hops_required"] >= 2

    def test_required_documents_exist_in_the_corpus(self):
        available = {p.stem for p in (ROOT / "data" / "documents").glob("*.md")}
        for q in self.questions:
            for doc_id in q.get("required_documents", []):
                assert doc_id in available, f"{q['id']} requires missing {doc_id}"


def test_strategy_registry_is_complete():
    from src.retrieval.strategies import STRATEGY_LABELS
    assert set(STRATEGIES) == set(STRATEGY_LABELS)
