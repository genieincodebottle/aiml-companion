"""End-to-end ingestion: files on disk -> a queryable knowledge graph.

Order matters, and every step depends on the one before it:

  1. schema        constraints and indexes first, so every MERGE below has an
                   index to seek on instead of a label scan
  2. backbone      ERP-style CSVs, which seed the resolver with canonical names
  3. chunk         structure-aware splitting
  4. extract       one LLM call per document -> entities + relationships
  5. embed         one vector per chunk, cached
  6. write         documents, chunks, entities, relationships, mentions
  7. verify        assert the graph is actually queryable before declaring done

Step 7 is the one people leave out.  An ingestion that reports success and
leaves an unusable graph behind is worse than one that fails, because the
failure surfaces later as bad answers rather than as an error.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Config, get_config
from ..guardrails import GuardrailEngine, GuardrailViolation
from ..graph import queries
from ..graph.client import GraphClient
from ..graph.schema import apply_schema, verify_vector_index
from ..llm import LLMClient
from .chunker import chunk_documents, load_documents
from .extract import GraphExtractor, link_mentions
from .loader import load_backbone
from .resolve import EntityResolver

log = logging.getLogger(__name__)

Progress = Callable[[str, float], None]


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    blocked_documents: list[dict[str, Any]] = field(default_factory=list)
    entities: int = 0
    relationships: int = 0
    mentions: int = 0
    backbone: dict[str, int] = field(default_factory=dict)
    resolver_stats: dict[str, int] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "blocked_documents": self.blocked_documents,
            "entities": self.entities,
            "relationships": self.relationships,
            "mentions": self.mentions,
            "backbone": self.backbone,
            "resolver": self.resolver_stats,
            "dropped_count": len(self.dropped),
            "usage": self.usage,
            "seconds": round(self.seconds, 1),
            "graph_counts": self.counts,
        }


def ingest(*, reset: bool = False, config: Config | None = None,
           progress: Progress | None = None) -> IngestReport:
    config = config or get_config()
    report = IngestReport()
    started = time.time()

    def step(message: str, fraction: float) -> None:
        log.info(message)
        if progress:
            progress(message, fraction)

    llm = LLMClient(config)
    resolver = EntityResolver()
    guard = GuardrailEngine(config)

    with GraphClient(config) as client:
        client.verify()

        if reset:
            step("Wiping existing graph", 0.02)
            client.wipe()

        # -- 1. schema ------------------------------------------------------
        step("Applying schema (constraints, indexes, vector index)", 0.05)
        dimensions = config.embedding["dimensions"]
        apply_schema(client, dimensions)
        verify_vector_index(client, dimensions)

        # -- 2. structured backbone ----------------------------------------
        step("Loading structured backbone from data/structured", 0.10)
        backbone = load_backbone(client, resolver, config.structured_dir)
        report.backbone = backbone.as_dict()

        # -- 3. chunk -------------------------------------------------------
        step("Loading and chunking documents", 0.15)
        documents = load_documents(config.documents_dir)
        chunking = config.chunking
        chunks = chunk_documents(
            documents,
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
            min_chunk_chars=chunking["min_chunk_chars"],
        )
        report.documents = len(documents)
        report.chunks = len(chunks)

        # -- 4. extract -----------------------------------------------------
        extractor = GraphExtractor(
            llm, resolver,
            entity_types=config.extraction["entity_types"],
            relation_types=config.extraction["relation_types"],
        )
        relations: list[Any] = []
        for i, doc in enumerate(documents):
            step(f"Extracting graph from {doc.doc_id} ({i + 1}/{len(documents)})",
                 0.15 + 0.45 * (i + 1) / len(documents))

            # THE GUARDRAIL THAT MATTERS MOST IN GRAPHRAG.
            #
            # This runs before the extractor, because the extractor's output is
            # written to shared persistent state. A poisoned document that gets
            # past this point does not corrupt one answer - it writes an edge
            # that every future traversal can reach, for every user, carrying a
            # real citation to a real sentence in a real document.
            try:
                decision = guard.check_document(doc.body, doc.doc_id)
            except GuardrailViolation as exc:
                log.error("BLOCKED %s: %s", doc.doc_id, exc)
                report.blocked_documents.append({
                    "doc_id": doc.doc_id, "kind": exc.kind, "reason": str(exc),
                })
                # Skip, do not abort. One hostile document in a batch of 500
                # must not stop the other 499 from being indexed, and the block
                # is recorded in the report and in the audit log either way.
                continue

            # Extract from the GUARDED text, not the original. Getting this
            # wrong - scanning one string and extracting from another - is a
            # guardrail that reports success and protects nothing.
            doc.body = decision.text
            result = extractor.extract_document(doc)
            relations.extend(result.relations)
            report.dropped.extend(result.dropped)

        # -- 5. embed -------------------------------------------------------
        step(f"Embedding {len(chunks)} chunks", 0.65)
        vectors = llm.embed_documents([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        # -- 6. write -------------------------------------------------------
        step("Writing documents and chunks", 0.75)
        client.run_batch(
            queries.UPSERT_DOCUMENTS,
            [
                {
                    "doc_id": d.doc_id, "title": d.title, "doc_type": d.doc_type,
                    "source_path": d.source_path, "published": d.published,
                }
                for d in documents
            ],
        )
        client.run_batch(
            queries.UPSERT_CHUNKS,
            [
                {
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "ord": c.ord,
                    "text": c.text, "embedding": c.embedding,
                }
                for c in chunks
            ],
            # Chunks carry a 768-float vector each, so the payload per row is
            # far larger than for a plain node.  A 500-row batch here is
            # megabytes; 100 keeps each transaction comfortable.
            batch_size=100,
        )

        step("Writing entities", 0.82)
        entities = resolver.all_entities()
        by_label: dict[str, list[dict[str, Any]]] = {}
        allowed = set(config.extraction["entity_types"])
        for entity in entities:
            if entity.type not in allowed:
                # Unreachable if the extractor validated, but this is the last
                # gate before a model-supplied string reaches a Cypher label.
                report.dropped.append(f"entity with unapproved label {entity.type}")
                continue
            by_label.setdefault(entity.type, []).append(
                {
                    "key": entity.key, "name": entity.name,
                    "aliases": entity.aliases, "type": entity.type,
                    "summary": entity.summary, "status": entity.status,
                }
            )
        for label, rows in by_label.items():
            client.run_batch(queries.upsert_entities(label), rows)
        report.entities = sum(len(r) for r in by_label.values())

        step("Writing relationships", 0.88)
        by_type: dict[str, list[dict[str, Any]]] = {}
        allowed_rels = set(config.extraction["relation_types"])
        for relation in relations:
            if relation.rel_type not in allowed_rels:
                report.dropped.append(f"relationship with unapproved type {relation.rel_type}")
                continue
            by_type.setdefault(relation.rel_type, []).append(
                {
                    "source": relation.source_key,
                    "target": relation.target_key,
                    "confidence": relation.confidence,
                    "source_doc": relation.source_doc,
                    "evidence": relation.evidence,
                    "provenance": "llm",
                }
            )
        for rel_type, rows in by_type.items():
            client.run_batch(queries.upsert_relationships(rel_type), rows)
        report.relationships = sum(len(r) for r in by_type.values())

        step("Linking chunk mentions", 0.94)
        mention_rows = link_mentions(chunks, entities)
        report.mentions = client.run_batch(queries.UPSERT_MENTIONS, mention_rows)

        # -- 7. verify ------------------------------------------------------
        step("Verifying the graph is queryable", 0.98)
        report.counts = client.counts()
        _verify(client, config, report)

    report.resolver_stats = resolver.stats
    report.usage = llm.usage.as_dict()
    report.seconds = time.time() - started
    step("Ingestion complete", 1.0)
    return report


def _verify(client: GraphClient, config: Config, report: IngestReport) -> None:
    """Post-ingest assertions.

    Each of these has a failure mode that is silent without the check, and each
    was chosen because it is the *earliest* observable symptom of a real bug.
    """
    problems: list[str] = []

    # A chunk without an embedding is invisible to vector search forever, and
    # nothing downstream will ever mention it.
    missing = client.run(
        "MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN count(c) AS n"
    )[0]["n"]
    if missing:
        problems.append(f"{missing} chunks have no embedding")

    # The vector index must actually return something.  If the index is empty
    # or mis-dimensioned, this is where you find out - not three days later.
    probe = client.run(
        queries.VECTOR_SEARCH,
        index_name="chunk_embedding_index",
        k=1,
        embedding=client.run(
            "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL "
            "RETURN c.embedding AS e LIMIT 1"
        )[0]["e"],
    )
    if not probe:
        problems.append(
            "the vector index returned no results for a vector taken from the "
            "index itself - the index is empty or its dimension does not match"
        )

    # No MENTIONS means the bridge between text and graph does not exist, and
    # every graph retrieval will silently degrade to nothing.
    if report.mentions == 0:
        problems.append("no MENTIONS relationships were created")

    # DEPENDS_ON is the whole reason this project exists.  Zero of them means
    # extraction ran but found no sub-tier structure.
    depends = report.counts.get("rel:DEPENDS_ON", 0)
    if depends == 0:
        problems.append("no DEPENDS_ON relationships were extracted")

    if problems:
        raise RuntimeError(
            "Ingestion finished but the graph is not usable:\n  - "
            + "\n  - ".join(problems)
        )
