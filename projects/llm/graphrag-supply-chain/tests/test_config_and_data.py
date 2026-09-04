"""Configuration guards and data-integrity checks on the shipped corpus.

The corpus is not decoration - the evaluation's claims rest on it having
specific properties. These tests assert those properties so a well-meaning edit
cannot quietly invalidate the benchmark.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from src.config import _is_placeholder

ROOT = Path(__file__).resolve().parent.parent
STRUCTURED = ROOT / "data" / "structured"


class TestPlaceholderDetection:
    """`.env.example` ships `GOOGLE_API_KEY=your-google-api-key-here`, and a
    placeholder is a perfectly truthy string. Checking "is it set?" lets setup
    pass and then fails 200 lines later as an opaque HTTP 400."""

    @pytest.mark.parametrize("value", [
        "your-google-api-key-here", "changeme", "xxx", "", "replace-me", "short",
    ])
    def test_placeholders_are_rejected(self, value):
        assert _is_placeholder(value)

    def test_a_real_looking_key_is_accepted(self):
        assert not _is_placeholder("AIzaSyD-8kQ2mVexampleKEY1234567890")


class TestConfigInvariants:
    def setup_method(self):
        with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as fh:
            self.cfg = yaml.safe_load(fh)

    def test_output_budgets_leave_room_for_reasoning(self):
        """Both output budgets must be large enough to hold reasoning AND output.

        On a thinking model, reasoning tokens come out of the same
        `max_output_tokens` budget as the visible response. Measured on this
        project, reasoning alone consumed ~1,900 tokens on a single answer.

        Both call sites were broken by this in turn, and neither failed loudly:

          - extraction at 2,048 returned truncated JSON and reported zero
            entities for EVERY document;
          - answering at 2,048 stopped after the first of four products, and the
            truncated answer read exactly like a complete one that had found
            less - which silently corrupted the benchmark built on it.

        This test asserts headroom rather than a relationship between the two
        numbers. An earlier version required extraction > answering, which was
        true at the time and encoded the wrong idea: the constraint is
        "big enough for reasoning plus output", not "one bigger than the other".
        """
        floor = 4096
        for key in ("max_output_tokens", "extraction_max_output_tokens"):
            assert self.cfg["llm"][key] >= floor, (
                f"llm.{key} is {self.cfg['llm'][key]}, under the {floor} floor. "
                "Reasoning tokens share this budget; too small truncates output "
                "silently."
            )

    def test_extraction_disables_thinking(self):
        """Extraction is transcription against a closed vocabulary with a schema
        already fixing the shape. There is nothing to deliberate about, and
        deliberation both costs reproducibility and starves the output."""
        assert self.cfg["llm"]["extraction_thinking_budget"] == 0

    def test_extraction_temperature_is_zero(self):
        # Extraction must be reproducible or the graph changes on every ingest.
        assert self.cfg["llm"]["temperature"] == 0.0

    def test_chunk_overlap_is_smaller_than_chunk_size(self):
        # Otherwise _split_long cannot advance and loops forever.
        assert self.cfg["chunking"]["chunk_overlap"] < self.cfg["chunking"]["chunk_size"]

    def test_every_extractable_relation_is_traversable(self):
        """If a type can be extracted but not traversed, the edge is written and
        then never used - invisible dead weight in the graph.

        Subset, not equality: CONTAINS is traversable but is never extracted,
        because the bill of materials comes from the PLM export rather than from
        prose. That asymmetry is the structured/unstructured split working as
        intended, so the test asserts the direction that matters.
        """
        from src.graph.queries import KNOWLEDGE_RELS
        extractable = set(self.cfg["extraction"]["relation_types"])
        assert extractable <= set(KNOWLEDGE_RELS), (
            f"extractable but not traversable: {extractable - set(KNOWLEDGE_RELS)}"
        )


def _read(name: str) -> list[dict]:
    with open(STRUCTURED / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestBackboneIntegrity:
    def test_bom_references_only_real_ids(self):
        products = {r["product_id"] for r in _read("products.csv")}
        components = {r["component_id"] for r in _read("components.csv")}
        for row in _read("bom.csv"):
            assert row["product_id"] in products
            assert row["component_id"] in components

    def test_supplies_references_only_real_ids(self):
        suppliers = {r["supplier_id"] for r in _read("suppliers.csv")}
        components = {r["component_id"] for r in _read("components.csv")}
        for row in _read("supplies.csv"):
            assert row["supplier_id"] in suppliers
            assert row["component_id"] in components

    def test_every_component_has_at_least_one_supplier(self):
        supplied = {r["component_id"] for r in _read("supplies.csv")}
        for row in _read("components.csv"):
            assert row["component_id"] in supplied, (
                f"{row['component_id']} has no supplier - it would be an orphan "
                "in every exposure traversal"
            )

    def test_sole_source_flags_are_consistent_with_share(self):
        """A component flagged sole_source must not also have a second supplier.

        This inconsistency would make the sole-source risk query lie, and it is
        exactly the kind of drift that creeps into a hand-maintained CSV.
        """
        by_component: dict[str, list[dict]] = {}
        for row in _read("supplies.csv"):
            by_component.setdefault(row["component_id"], []).append(row)
        for component, rows in by_component.items():
            sole = [r for r in rows if r["sole_source"].lower() == "true"]
            if sole:
                assert len(rows) == 1, (
                    f"{component} is flagged sole_source but has {len(rows)} suppliers"
                )

    def test_dual_sourced_shares_sum_to_100(self):
        by_component: dict[str, int] = {}
        for row in _read("supplies.csv"):
            by_component[row["component_id"]] = (
                by_component.get(row["component_id"], 0) + int(row["share_pct"])
            )
        for component, total in by_component.items():
            assert total == 100, f"{component} shares sum to {total}, not 100"


class TestCorpusProperties:
    """Properties the evaluation's claims depend on."""

    def test_corpus_is_large_enough_for_top_k_to_be_a_real_filter(self):
        """A tiny corpus cannot demonstrate anything about retrieval.

        This is the trap that invalidates a lot of published RAG comparisons: on
        a four-document corpus, top-5 retrieval returns essentially the whole
        thing, so every strategy sees the same context and the resulting table
        measures nothing but noise.

        The number that matters is not bytes, it is CHUNKS PER RETRIEVAL SLOT.
        With a top_k of 5, a 151-chunk corpus means retrieval discards about 97%
        of what it could have returned, so which 5 it picks is a real decision.
        """
        docs = list((ROOT / "data" / "documents").glob("*.md"))
        assert len(docs) >= 30

        from src.ingest.chunker import chunk_documents, load_documents
        chunks = chunk_documents(
            load_documents(ROOT / "data" / "documents"),
            chunk_size=900, chunk_overlap=150, min_chunk_chars=120,
        )
        with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as fh:
            top_k = yaml.safe_load(fh)["retrieval"]["vector_top_k"]
        assert len(chunks) >= 20 * top_k, (
            f"{len(chunks)} chunks against top_k={top_k} is too few for the "
            "retrieval comparison to be meaningful"
        )

    def test_the_flagship_answer_appears_in_no_single_document(self):
        """The core claim of the project, asserted against the actual files.

        If any one document ever comes to contain both a Kaohsiung tier-2
        supplier and the finished products it ultimately feeds, the multi-hop
        question becomes a lookup and the headline comparison is invalid.
        """
        products = {"NW-500", "NW-220", "TX-9", "AQ-100"}
        for path in (ROOT / "data" / "documents").glob("*.md"):
            text = path.read_text(encoding="utf-8")
            if "Formosa Substrate Materials" not in text:
                continue
            named = {p for p in products if p in text}
            assert not named, (
                f"{path.name} names Formosa Substrate Materials AND products "
                f"{named}. The flagship question is no longer multi-hop."
            )
