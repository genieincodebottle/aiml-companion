"""LLM extraction of entities and relationships from document text.

This is where unstructured text becomes a graph, and it is the step with the
most ways to go wrong.  Four decisions, each argued.

DECISION 1: extract per DOCUMENT, not per chunk.

The obvious design runs the extractor on every chunk.  It is wrong here for two
reasons.  First, cost: this corpus is 33 documents and roughly 200 chunks, so
per-chunk extraction is 6x the calls for the same facts.  Second and more
important, correctness: relationships in these documents routinely span
sections.  A supplier profile names the company in its heading and the sub-tier
dependency four paragraphs later.  A chunk-level extractor reads
"Formosa Substrate Materials is its only qualified laminate source" with no
idea whose source it is, and either invents a subject or emits nothing.

The cost of the document-level choice is a context-length ceiling.  At 33 short
documents it is nowhere near binding.  For a corpus of 400-page contracts you
would have to go back to a windowed approach with a carried-forward summary,
and that is noted in docs/production-notes.md.

DECISION 2: mentions are linked WITHOUT the model.

We still need (:Chunk)-[:MENTIONS]->(:Entity) at chunk granularity, because
that edge is what turns a vector hit into graph anchors.  But asking a model
which entities a chunk mentions is paying for string matching.  Once the
document-level pass has told us which entities exist and under which aliases,
finding them in each chunk is exact, free, instant and reproducible.  The
model is used for the part that needs judgement and not for the part that
needs a substring search.

DECISION 3: a closed vocabulary, enforced twice.

The schema constrains types, and `_validate` drops anything outside the
configured lists anyway.  Belt and braces, because a model that invents the
label "SubTierSupplier" once in fifty documents produces a node nothing will
ever traverse to, and no error anywhere.

DECISION 4: every relationship carries its evidence span.

The extractor must return the sentence it based each edge on, and that sentence
is stored on the relationship.  This is the difference between a knowledge
graph and a rumour graph.  It costs output tokens; it buys the ability to show
a human why the system believes Meridian depends on Formosa, and to find out
fast when it is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from .chunker import Chunk, Document
from .resolve import EntityResolver, ResolvedEntity, normalise

log = logging.getLogger(__name__)


@dataclass
class ExtractedRelation:
    source_key: str
    target_key: str
    rel_type: str
    confidence: float
    evidence: str
    source_doc: str


@dataclass
class ExtractionResult:
    entities: list[ResolvedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """\
You extract a supply chain knowledge graph from corporate documents for a \
medical device manufacturer called Northwind Instruments.

Rules you must follow:

1. Extract ONLY what the document states. Never infer a relationship from \
general knowledge, from company names, or from what would be plausible. If the \
document does not say it, it does not exist.
2. Northwind Instruments is the document author, not a supplier. Never create \
an entity for Northwind Instruments itself.
3. For every relationship, quote the exact sentence from the document that \
supports it in the `evidence` field. Copy it verbatim. If you cannot quote a \
sentence, do not emit the relationship.
4. Use the document's own naming. Do not expand abbreviations or tidy up \
company names; alias handling happens downstream.
5. `confidence` reflects how explicitly the document states the relationship: \
1.0 for a direct statement, 0.7 for a clearly implied one, below 0.5 for \
anything you are unsure about.
6. A document describing a supplier's own purchases creates DEPENDS_ON from \
the buying supplier to the selling supplier, and SUPPLIES from the seller to \
the material or component sold.
"""

USER_TEMPLATE = """\
Allowed entity types: {entity_types}
Allowed relationship types: {relation_types}

Relationship semantics:
  DEPENDS_ON      (Supplier) depends on (Supplier)      buyer -> seller
  SUPPLIES        (Supplier) supplies (Component|Material)
  OPERATES        (Supplier) operates (Site)
  LOCATED_IN      (Site|Supplier) is located in (Location). Use Site when the
                  document names a specific facility, and Supplier directly
                  when it only says the company is in a place.
  AFFECTS         (Incident) affects (Site|Supplier|Location)
  RAISED_AGAINST  (Finding) was raised against (Supplier)
  HOLDS           (Supplier) holds (Certification)
  APPLIES_TO      (Regulation) applies to (Component|Material)

Document title: {title}
Document type: {doc_type}
Published: {published}

--- DOCUMENT START ---
{body}
--- DOCUMENT END ---

Extract the entities and relationships this document states."""


# The schema handed to the model.  Structured output turns "please reply in
# JSON" from a request into a constraint the decoder enforces.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "summary": {
                        "type": "string",
                        "description": "One clause describing this entity, from the document only.",
                    },
                    "status": {
                        "type": "string",
                        "description": (
                            "For Finding entities only: 'open' if the document "
                            "says the finding or corrective action is still "
                            "open, 'closed' if it was closed. Empty for every "
                            "other entity type."
                        ),
                    },
                },
                "required": ["name", "type"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Entity name, exactly as listed in entities."},
                    "source_type": {"type": "string"},
                    "target": {"type": "string"},
                    "target_type": {"type": "string"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string", "description": "Verbatim sentence from the document."},
                },
                "required": ["source", "source_type", "target", "target_type",
                             "type", "confidence", "evidence"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


class GraphExtractor:
    def __init__(self, llm: LLMClient, resolver: EntityResolver,
                 entity_types: list[str], relation_types: list[str]) -> None:
        self.llm = llm
        self.resolver = resolver
        self.entity_types = entity_types
        self.relation_types = relation_types
        # Case-insensitive lookup so "supplier" and "Supplier" both validate,
        # while the canonical casing is what reaches the database.
        self._entity_lookup = {t.lower(): t for t in entity_types}
        self._relation_lookup = {t.lower(): t for t in relation_types}

    def extract_document(self, doc: Document) -> ExtractionResult:
        prompt = USER_TEMPLATE.format(
            entity_types=", ".join(self.entity_types),
            relation_types=", ".join(self.relation_types),
            title=doc.title,
            doc_type=doc.doc_type,
            published=doc.published,
            body=doc.body,
        )
        payload = self.llm.extract_json(
            prompt, EXTRACTION_SCHEMA, system=SYSTEM_PROMPT
        )
        if not payload:
            log.warning("no extraction returned for %s", doc.doc_id)
            return ExtractionResult()
        return self._validate(payload, doc)

    def _validate(self, payload: dict[str, Any], doc: Document) -> ExtractionResult:
        result = ExtractionResult()

        # Pass 1: entities.  Resolution happens here, so by the time we look at
        # relationships every name already has a canonical key.
        local: dict[str, ResolvedEntity] = {}
        for raw in payload.get("entities") or []:
            name = (raw.get("name") or "").strip()
            etype = self._entity_lookup.get((raw.get("type") or "").strip().lower())
            if not name or not etype:
                result.dropped.append(
                    f"entity '{name}' with unknown type '{raw.get('type')}'"
                )
                continue
            entity = self.resolver.resolve(
                etype, name, summary=(raw.get("summary") or "").strip(),
                status=(raw.get("status") or "").strip().lower(),
            )
            local[normalise(name)] = entity
            result.entities.append(entity)

        # Pass 2: relationships.
        for raw in payload.get("relationships") or []:
            rtype = self._relation_lookup.get((raw.get("type") or "").strip().lower())
            if not rtype:
                result.dropped.append(f"relationship of unknown type '{raw.get('type')}'")
                continue

            source = self._lookup(local, raw.get("source"), raw.get("source_type"))
            target = self._lookup(local, raw.get("target"), raw.get("target_type"))
            if source is None or target is None:
                # The model referenced a name it did not list as an entity.
                # Dropping is right: creating the node here would mean creating
                # an entity nothing validated, from a field nothing checked.
                result.dropped.append(
                    f"{rtype} between unlisted entities "
                    f"'{raw.get('source')}' -> '{raw.get('target')}'"
                )
                continue

            if source.key == target.key:
                result.dropped.append(f"self-loop {rtype} on {source.name}")
                continue

            evidence = (raw.get("evidence") or "").strip()
            if not evidence:
                result.dropped.append(f"{rtype} {source.name}->{target.name} without evidence")
                continue

            result.relations.append(
                ExtractedRelation(
                    source_key=source.key,
                    target_key=target.key,
                    rel_type=rtype,
                    confidence=_clamp(raw.get("confidence")),
                    evidence=evidence[:600],
                    source_doc=doc.doc_id,
                )
            )
        return result

    def _lookup(self, local: dict[str, ResolvedEntity], name: Any,
                etype: Any) -> ResolvedEntity | None:
        if not name:
            return None
        hit = local.get(normalise(str(name)))
        if hit:
            return hit
        # Not in this document's entity list, but it may be a known entity from
        # the ERP backbone or an earlier document.  Resolve against the global
        # registry - but only if the declared type is valid.
        canonical = self._entity_lookup.get((str(etype) or "").strip().lower())
        if not canonical:
            return None
        resolved = self.resolver.resolve(canonical, str(name))
        return resolved


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def link_mentions(chunks: list[Chunk], entities: list[ResolvedEntity],
                  ) -> list[dict[str, Any]]:
    """Build (:Chunk)-[:MENTIONS]->(:Entity) rows by string matching.

    No model call.  For each entity we test its canonical name and every alias
    against the normalised chunk text.

    The confidence recorded reflects *how* it matched, and the distinction is
    real: a full canonical name appearing in text is near-certain, while a
    short alias like "Volta" is more likely to be a coincidence, so it is
    recorded lower and can be filtered out of traversal by
    ``extraction.min_confidence``.
    """
    rows: list[dict[str, Any]] = []
    # Pre-normalise once.  Doing it inside the loop is O(chunks x entities)
    # normalisations, which on a real corpus is minutes of pure waste.
    surfaces: list[tuple[str, str, float]] = []
    for entity in entities:
        surfaces.append((entity.key, normalise(entity.name), 1.0))
        for alias in entity.aliases:
            norm = normalise(alias)
            if not norm:
                continue
            # Short aliases are ambiguous by nature; record them as weaker.
            confidence = 0.9 if len(norm) >= 12 else 0.6
            surfaces.append((entity.key, norm, confidence))

    for chunk in chunks:
        chunk_norm = normalise(chunk.text)
        best: dict[str, float] = {}
        for key, surface, confidence in surfaces:
            if surface and surface in chunk_norm:
                best[key] = max(best.get(key, 0.0), confidence)
        for key, confidence in best.items():
            rows.append(
                {"chunk_id": chunk.chunk_id, "key": key, "confidence": confidence}
            )
    return rows
