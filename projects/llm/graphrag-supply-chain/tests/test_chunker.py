"""Chunking tests.

The failure these guard against is the one that is hardest to trace: a chunk
boundary through the middle of a relationship statement means the extractor
never sees the relationship, so the edge is never created, so the multi-hop
query returns nothing - and the symptom appears three layers away from the
cause, in the retrieval results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.chunker import (Document, chunk_document, load_documents,
                                _sections, _split_long)

CORPUS = Path(__file__).resolve().parent.parent / "data" / "documents"

DEFAULTS = {"chunk_size": 900, "chunk_overlap": 150, "min_chunk_chars": 120}


def _doc(body: str) -> Document:
    return Document(doc_id="TEST-1", title="Test Document", doc_type="test",
                    published="2026-01-01", body=body, source_path="test.md")


class TestSectioning:
    def test_preamble_before_first_heading_is_kept(self):
        # In this corpus the opening summary lives here and is often the most
        # quotable sentence in the file. Dropping it is silent content loss.
        sections = _sections("Opening summary line.\n\n# Heading\n\nBody.")
        assert sections[0][1].startswith("Opening summary")

    def test_headings_become_boundaries(self):
        sections = _sections("# A\n\nAlpha.\n\n# B\n\nBeta.")
        assert [s[0] for s in sections] == ["A", "B"]


class TestSplitting:
    def test_short_text_is_not_split(self):
        assert _split_long("short", 900, 150) == ["short"]

    def test_long_text_breaks_on_a_separator_not_mid_word(self):
        text = ("Meridian purchases the copper-clad laminate from Formosa. " * 40)
        pieces = _split_long(text, 300, 60)
        assert len(pieces) > 1
        for piece in pieces:
            # No piece may begin or end inside a word.
            assert piece == piece.strip()
            assert not piece.startswith("aminate")

    def test_overlap_preserves_straddling_statements(self):
        text = "A" * 200 + " the critical relationship sentence here " + "B" * 200
        pieces = _split_long(text, 220, 120)
        joined = " ".join(pieces)
        assert "critical relationship sentence" in joined


class TestChunkDocument:
    def test_chunk_ids_are_deterministic(self):
        """Re-running ingestion on unchanged input must produce identical ids.

        MERGE keys on chunk_id, so a random id here would double the graph on
        every re-ingest instead of updating it.
        """
        doc = _doc("# A\n\n" + "Body text. " * 50)
        first = [c.chunk_id for c in chunk_document(doc, **DEFAULTS)]
        second = [c.chunk_id for c in chunk_document(doc, **DEFAULTS)]
        assert first == second
        assert first[0] == "TEST-1::000"

    def test_every_chunk_carries_its_document_title(self):
        """Contextual retrieval. A chunk beginning "Corrective action requested"
        is unattributable on its own, to the embedder and to the reader."""
        doc = _doc("# Findings\n\n" + "Corrective action requested. " * 20)
        for chunk in chunk_document(doc, **DEFAULTS):
            assert "Test Document" in chunk.text

    def test_tiny_sections_are_merged_not_dropped(self):
        doc = _doc("# A\n\n" + "x" * 400 + "\n\n# B\n\nToo short.")
        chunks = chunk_document(doc, **DEFAULTS)
        assert "Too short." in " ".join(c.text for c in chunks)


class TestRealCorpus:
    """Run against the shipped corpus, because a chunker that works on
    synthetic input and not on the real documents is not working."""

    def test_corpus_loads(self):
        docs = load_documents(CORPUS)
        assert len(docs) >= 30
        assert all(d.doc_id and d.title and d.doc_type for d in docs)

    def test_document_ids_are_unique(self):
        docs = load_documents(CORPUS)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_front_matter_id_matches_filename(self):
        # Citations are built from doc_id. If it drifts from the filename,
        # tracing a citation back to a file becomes guesswork.
        for doc in load_documents(CORPUS):
            assert doc.source_path == f"{doc.doc_id}.md"

    def test_key_relationship_sentences_survive_chunking_intact(self):
        """The load-bearing test.

        Each of these sentences states a sub-tier dependency that the whole
        project depends on extracting. If chunking splits one, the edge is lost
        and the flagship query silently returns less exposure than it should.
        """
        required = [
            ("SUP-PROFILE-MERIDIAN", "purchased from Formosa Substrate Materials"),
            ("SUP-PROFILE-VOLTA", "from Baltic Lithium Salts"),
            ("SUP-PROFILE-NORDCELL", "from Baltic Lithium Salts"),
            ("SUB-TIER-FORMOSA", "Sarawak Copper Foil"),
            ("SUP-PROFILE-HELIOS", "from Anhui Rare Earth Refining"),
            ("SUP-PROFILE-KAIGAN", "from Kaohsiung Precision Glass"),
        ]
        docs = {d.doc_id: d for d in load_documents(CORPUS)}
        for doc_id, phrase in required:
            chunks = chunk_document(docs[doc_id], **DEFAULTS)
            # Collapse whitespace before searching. The source documents are
            # hard-wrapped at 78 columns, so a sentence routinely contains a
            # newline between two words. Searching for a literal single-space
            # phrase would fail on formatting rather than on chunking, which is
            # a test that reports the wrong problem.
            flat = [" ".join(c.text.split()) for c in chunks]
            assert any(phrase in text for text in flat), (
                f"'{phrase}' was split across chunk boundaries in {doc_id}. "
                "The extractor will not see this relationship."
            )
