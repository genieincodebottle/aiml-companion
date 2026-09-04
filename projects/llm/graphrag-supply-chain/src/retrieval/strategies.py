"""The four retrieval strategies, behind one interface.

  vector    Dense similarity only.  The textbook RAG baseline.
  keyword   BM25 only.  Rarely used alone; here so the comparison can show
            what each half of the classic baseline contributes.
  classic   vector + BM25 fused with RRF.  This is the *honest* baseline - the
            thing a competent team would actually ship before reaching for a
            graph.  Comparing GraphRAG against dense-only would be rigging the
            experiment.
  graph     Entity linking -> traversal -> derived facts -> supporting text.
            No vector search at all.
  hybrid    Vector anchors -> MENTIONS -> traversal -> derived facts, fused
            with the vector and keyword hits.  The full architecture.

Every strategy returns a RetrievalResult with the same shape, records the
Cypher it ran, and narrates its own steps into ``trace`` so the UI can show a
learner exactly what happened rather than asserting that something did.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..config import Config
from ..graph import queries
from ..graph.client import GraphClient
from ..graph.schema import (CHUNK_FULLTEXT_INDEX, ENTITY_FULLTEXT_INDEX,
                            VECTOR_INDEX_NAME)
from ..llm import LLMClient
from .base import Evidence, GraphEntity, RetrievalResult, reciprocal_rank_fusion

STRATEGIES = ["vector", "keyword", "classic", "graph", "hybrid"]

STRATEGY_LABELS = {
    "vector": "Vector only (dense RAG)",
    "keyword": "Keyword only (BM25)",
    "classic": "Classic RAG (vector + BM25)",
    "graph": "GraphRAG (traversal only)",
    "hybrid": "Hybrid GraphRAG (vector + graph)",
}

# Lucene reserved characters.  A question containing "NW-500" is a syntax error
# to the Lucene parser if passed raw, and the driver surfaces it as an opaque
# server exception - so every full-text query goes through _lucene() first.
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "at",
    "to", "for", "and", "or", "our", "we", "do", "does", "did", "if", "it",
    "that", "this", "these", "those", "be", "been", "has", "have", "had",
    "would", "could", "should", "any", "all", "from", "by", "with", "about",
}


def _lucene(text: str, *, min_len: int = 2) -> str:
    """Turn a natural-language question into a safe Lucene OR query.

    Stopwords are dropped because Lucene will happily match "the" in every
    chunk in the corpus and drown the signal.  Everything else is escaped, not
    stripped: "NW-500" must survive as a searchable token, since exact
    identifier match is the entire reason the keyword arm exists.
    """
    tokens = []
    for raw in re.split(r"\s+", text.strip()):
        token = _LUCENE_SPECIAL.sub(r"\\\1", raw).strip()
        bare = re.sub(r"[^\w]", "", token).lower()
        if len(bare) < min_len or bare in _STOPWORDS:
            continue
        tokens.append(token)
    return " OR ".join(tokens) if tokens else _LUCENE_SPECIAL.sub(r"\\\1", text.strip())


def _to_evidence(rows: list[dict[str, Any]], retrieved_by: str) -> list[Evidence]:
    return [
        Evidence(
            kind="text",
            text=row["text"],
            source_id=row["chunk_id"],
            doc_id=row.get("doc_id", ""),
            title=row.get("title", ""),
            doc_type=row.get("doc_type", ""),
            score=float(row.get("score") or 0.0),
            retrieved_by=retrieved_by,
            detail={k: row[k] for k in ("mention_count", "entities") if k in row},
        )
        for row in rows
    ]


class Retriever:
    """Runs any strategy against one graph.  Stateless between calls apart from
    the clients it holds, so a single instance is safe to reuse - which matters
    in Streamlit, where the alternative is a new Neo4j connection pool on every
    keystroke."""

    def __init__(self, client: GraphClient, llm: LLMClient, config: Config) -> None:
        self.client = client
        self.llm = llm
        self.config = config
        self.settings = config.retrieval
        self.min_confidence = config.extraction["min_confidence"]

    # ------------------------------------------------------------------ entry
    def retrieve(self, question: str, strategy: str) -> RetrievalResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}'; expected one of {STRATEGIES}")
        started = time.perf_counter()
        result = getattr(self, f"_{strategy}")(question)
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------- primitives
    def _vector_rows(self, question: str, k: int) -> list[dict[str, Any]]:
        embedding = self.llm.embed_query(question)
        return self.client.run(
            queries.VECTOR_SEARCH,
            index_name=VECTOR_INDEX_NAME, k=k, embedding=embedding,
        )

    def _keyword_rows(self, question: str, k: int) -> list[dict[str, Any]]:
        return self.client.run(
            queries.FULLTEXT_CHUNK_SEARCH,
            index_name=CHUNK_FULLTEXT_INDEX, query=_lucene(question), k=k,
        )

    # ------------------------------------------------------------- strategies
    def _vector(self, question: str) -> RetrievalResult:
        k = self.settings["vector_top_k"]
        rows = self._vector_rows(question, k)
        result = RetrievalResult(strategy="vector")
        result.evidence = _to_evidence(rows, "vector")
        result.cypher_run.append(queries.VECTOR_SEARCH)
        result.trace.append(
            f"Embedded the question and asked the vector index for the {k} "
            f"nearest chunks. Got {len(rows)}."
        )
        result.trace.append(
            "No entity linking, no traversal. Whatever the answer needs must be "
            "inside these chunks, because nothing else was retrieved."
        )
        return result

    def _keyword(self, question: str) -> RetrievalResult:
        k = self.settings["vector_top_k"]
        rows = self._keyword_rows(question, k)
        result = RetrievalResult(strategy="keyword")
        result.evidence = _to_evidence(rows, "keyword")
        result.cypher_run.append(queries.FULLTEXT_CHUNK_SEARCH)
        result.trace.append(f"BM25 over chunk text via Lucene. Got {len(rows)} chunks.")
        return result

    def _classic(self, question: str) -> RetrievalResult:
        k = self.settings["vector_top_k"]
        # Over-fetch each arm, then fuse down to k.  Fusing two lists of length
        # k and keeping k means each arm effectively contributes k/2, which
        # makes the hybrid *worse* than either alone on a question one arm
        # answers cleanly.  Over-fetching is what makes fusion additive.
        vector_rows = self._vector_rows(question, k * 2)
        keyword_rows = self._keyword_rows(question, k * 2)
        result = RetrievalResult(strategy="classic")
        result.evidence = reciprocal_rank_fusion(
            {
                "vector": _to_evidence(vector_rows, "vector"),
                "keyword": _to_evidence(keyword_rows, "keyword"),
            },
            limit=k,
        )
        result.cypher_run.extend([queries.VECTOR_SEARCH, queries.FULLTEXT_CHUNK_SEARCH])
        result.trace.append(
            f"Vector search returned {len(vector_rows)} candidates, BM25 returned "
            f"{len(keyword_rows)}."
        )
        result.trace.append(
            f"Fused with Reciprocal Rank Fusion (k={60}) and kept the top {k}."
        )
        result.stats["vector_candidates"] = len(vector_rows)
        result.stats["keyword_candidates"] = len(keyword_rows)
        return result

    def _graph(self, question: str) -> RetrievalResult:
        result = RetrievalResult(strategy="graph")
        seeds = self._link_entities(question, result)
        if not seeds:
            result.trace.append(
                "No entity in the question matched anything in the graph, so "
                "there is nowhere to start traversing. This is the honest "
                "failure mode of graph-only retrieval and it is why the hybrid "
                "strategy keeps a vector arm."
            )
            return result
        self._expand_and_gather(seeds, result)
        return result

    def _hybrid(self, question: str) -> RetrievalResult:
        result = RetrievalResult(strategy="hybrid")
        k = self.settings["vector_top_k"]
        anchor_k = self.settings["graph_anchor_top_k"]

        vector_rows = self._vector_rows(question, k * 2)
        keyword_rows = self._keyword_rows(question, k * 2)
        result.cypher_run.extend([queries.VECTOR_SEARCH, queries.FULLTEXT_CHUNK_SEARCH])
        result.trace.append(
            f"Vector search returned {len(vector_rows)} chunks and BM25 "
            f"{len(keyword_rows)}."
        )

        # Two independent sources of graph anchors, because each fails
        # differently.  Name matching fails when the question never names an
        # entity ("which products are exposed to the typhoon"). Chunk-mention
        # anchoring fails when vector search lands on the wrong document.
        # Together they are considerably more robust than either.
        seeds = self._link_entities(question, result)

        anchor_ids = [row["chunk_id"] for row in vector_rows[:anchor_k]]
        if anchor_ids:
            mention_rows = self.client.run(
                queries.CHUNK_ENTITIES,
                chunk_ids=anchor_ids, min_confidence=self.min_confidence,
            )
            result.cypher_run.append(queries.CHUNK_ENTITIES)
            from_chunks = [row["key"] for row in mention_rows]
            new = [key for key in from_chunks if key not in seeds]
            seeds.extend(new)
            result.trace.append(
                f"Walked MENTIONS from the top {len(anchor_ids)} vector hits and "
                f"picked up {len(new)} further anchor entities "
                f"({', '.join(r['name'] for r in mention_rows[:6])}"
                f"{'...' if len(mention_rows) > 6 else ''})."
            )

        text_lists = {
            "vector": _to_evidence(vector_rows, "vector"),
            "keyword": _to_evidence(keyword_rows, "keyword"),
        }

        if seeds:
            graph_result = RetrievalResult(strategy="graph")
            self._expand_and_gather(seeds, graph_result)
            result.entities = graph_result.entities
            result.cypher_run.extend(graph_result.cypher_run)
            result.trace.extend(graph_result.trace)
            # Graph facts bypass fusion entirely.  They are not competing with
            # the text for a slot: they are a different kind of evidence that
            # the text cannot supply, and ranking them against cosine scores
            # would be comparing a derived join to a similarity.
            result.evidence.extend(graph_result.graph_evidence)
            text_lists["graph"] = graph_result.text_evidence

        result.evidence.extend(reciprocal_rank_fusion(text_lists, limit=k))
        result.trace.append(
            f"Fused {len(text_lists)} ranked text lists with RRF and kept {k}, "
            f"then prepended {len(result.graph_evidence)} derived graph facts."
        )
        return result

    # ---------------------------------------------------------------- helpers
    def _link_entities(self, question: str, result: RetrievalResult) -> list[str]:
        rows = self.client.run(
            queries.ENTITY_LINK,
            index_name=ENTITY_FULLTEXT_INDEX, query=_lucene(question), k=8,
        )
        result.cypher_run.append(queries.ENTITY_LINK)
        if not rows:
            return []
        # Lucene scores are unbounded, so an absolute cutoff is meaningless.
        # A relative one is not: keep anything within half the top score, which
        # drops the long tail of one-token coincidental matches without
        # hardcoding a number that only suits this corpus.
        best = rows[0]["score"]
        kept = [row for row in rows if row["score"] >= best * 0.5]
        result.trace.append(
            "Linked the question to graph entities by full-text name match: "
            + ", ".join(f"{r['name']} ({r['type']})" for r in kept)
        )
        return [row["key"] for row in kept]

    def _expand_and_gather(self, seeds: list[str], result: RetrievalResult) -> None:
        """Traverse from the seed entities, run the type-driven templates, and
        pull back the text evidence behind everything found."""
        hops = self.settings["max_hops"]
        limit = self.settings["neighbour_chunk_limit"]

        neighbours = self.client.run(
            queries.neighbourhood(hops),
            keys=seeds, min_confidence=self.min_confidence, limit=60,
        )
        result.cypher_run.append(queries.neighbourhood(hops))
        result.entities = [
            GraphEntity(
                key=row["key"], name=row["name"], type=row["type"],
                hops=row["hops"], path_names=row["path_names"] or [],
                path_rels=row["path_rels"] or [],
            )
            for row in neighbours
        ]
        result.trace.append(
            f"Expanded up to {hops} hops from {len(seeds)} seed entities and "
            f"reached {len(neighbours)} further entities."
        )

        self._run_templates(seeds, result)

        # Pull text evidence for the seeds AND the reached entities.  This is
        # the return leg of the bridge: entities the traversal discovered become
        # chunks, and those chunks are frequently ones no vector search would
        # have ranked anywhere near the top.
        all_keys = seeds + [e.key for e in result.entities]
        chunk_rows = self.client.run(
            queries.CHUNKS_FOR_ENTITIES,
            keys=all_keys, min_confidence=self.min_confidence, limit=limit,
        )
        result.cypher_run.append(queries.CHUNKS_FOR_ENTITIES)
        result.evidence.extend(_to_evidence(chunk_rows, "graph"))
        result.trace.append(
            f"Pulled {len(chunk_rows)} supporting chunks for those entities via "
            "MENTIONS."
        )
        result.stats["seed_entities"] = len(seeds)
        result.stats["reached_entities"] = len(result.entities)

    def _run_templates(self, seeds: list[str], result: RetrievalResult) -> None:
        """Deterministic, type-driven query planning.

        Which template runs is decided by the TYPE of the linked entity, never
        by keyword-matching the question.  A Location anchor means the exposure
        question is on the table whatever words were used to ask it.

        The alternative - having the LLM write Cypher (text-to-Cypher) - is more
        flexible and considerably less safe: it can generate an unbounded
        traversal, it fails in ways that are hard to detect because a wrong
        query still returns rows, and it puts a model in the position of
        deciding what the database does. Templates cover the questions this
        domain actually gets, and they cannot surprise you. The trade-off is
        discussed in docs/production-notes.md.
        """
        rows = self.client.run(queries.ENTITIES_BY_KEY, keys=seeds)
        limit = 40

        for row in rows:
            key, name, etype = row["key"], row["name"], row["type"]

            if etype == "Location":
                exposure = self.client.run(
                    queries.PRODUCTS_EXPOSED_TO_LOCATION,
                    location_key=key, limit=limit,
                )
                result.cypher_run.append(queries.PRODUCTS_EXPOSED_TO_LOCATION)
                if exposure:
                    result.evidence.append(
                        Evidence(
                            kind="graph_fact",
                            source_id=f"exposure::{key}",
                            text=_render_exposure(name, exposure),
                            retrieved_by="graph",
                            detail={"rows": exposure},
                        )
                    )
                    result.trace.append(
                        f"'{name}' is a Location, so ran the exposure traversal: "
                        f"{len(exposure)} product/component paths pass through it."
                    )

            elif etype == "Supplier":
                impact = self.client.run(
                    queries.SUPPLIER_DOWNSTREAM_IMPACT, key=key, limit=limit,
                )
                result.cypher_run.append(queries.SUPPLIER_DOWNSTREAM_IMPACT)
                if impact:
                    result.evidence.append(
                        Evidence(
                            kind="graph_fact", source_id=f"impact::{key}",
                            text=_render_impact(name, impact),
                            retrieved_by="graph", detail={"rows": impact},
                        )
                    )
                    depth = max(row["tier_depth"] for row in impact)
                    result.trace.append(
                        f"'{name}' is a Supplier, so traced downstream impact: "
                        f"{len(impact)} product paths, deepest at tier {depth}."
                    )

            elif etype == "Product":
                tree = self.client.run(queries.PRODUCT_SUPPLY_TREE, key=key, limit=limit)
                result.cypher_run.append(queries.PRODUCT_SUPPLY_TREE)
                if tree:
                    result.evidence.append(
                        Evidence(
                            kind="graph_fact", source_id=f"tree::{key}",
                            text=_render_product_tree(name, tree),
                            retrieved_by="graph", detail={"rows": tree},
                        )
                    )
                    result.trace.append(
                        f"'{name}' is a Product, so pulled its full supply tree "
                        f"({len(tree)} component rows)."
                    )

            elif etype == "Component":
                chain = self.client.run(queries.COMPONENT_SUPPLY_CHAIN, key=key, limit=limit)
                shared = self.client.run(queries.SHARED_UPSTREAM_FOR_COMPONENT, key=key)
                result.cypher_run.extend(
                    [queries.COMPONENT_SUPPLY_CHAIN, queries.SHARED_UPSTREAM_FOR_COMPONENT]
                )
                if chain:
                    result.evidence.append(
                        Evidence(
                            kind="graph_fact", source_id=f"chain::{key}",
                            text=_render_component_chain(name, chain, shared),
                            retrieved_by="graph",
                            detail={"chain": chain, "shared_upstream": shared},
                        )
                    )
                    result.trace.append(
                        f"'{name}' is a Component, so pulled its supply chain and "
                        f"tested whether its suppliers share an upstream source "
                        f"({len(shared)} shared upstream found)."
                    )


# ---------------------------------------------------------------------------
# Rendering derived facts into text the model can read.
#
# These are deliberately plain and tabular.  A derived fact is not prose and
# should not pretend to be: presenting it as a small table makes it obvious to
# the model - and to the reader checking the answer - which parts came from a
# traversal rather than from a document.
# ---------------------------------------------------------------------------
def _render_exposure(location: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"Exposure traversal for location '{location}'.",
        "Each row is a path from a supplier operating in this location through "
        "to a finished product. tier_depth 0 means the exposed supplier sells "
        "to us directly; 1 or more means the exposure reaches us through that "
        "many intermediate suppliers.",
        "",
    ]
    for row in rows:
        chain = " -> ".join(reversed(row.get("dependency_chain") or []))
        lines.append(
            f"- product={row['product']} | component={row['component']} | "
            f"direct_supplier={row['direct_supplier']} | "
            f"exposed_supplier={row['exposed_supplier']} | "
            f"site={row['exposed_site']} | tier_depth={row['tier_depth']} | "
            f"sole_source={row['sole_source']}"
            + (f" | chain={chain}" if row["tier_depth"] else "")
        )
    return "\n".join(lines)


def _render_impact(supplier: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"Downstream impact traversal for supplier '{supplier}'.",
        "Each row is a path from this supplier through to a finished product. "
        "tier_depth 0 means they sell to Northwind directly; 1 or more means "
        "the dependency reaches us through that many intermediate suppliers, "
        "and Northwind has no direct commercial relationship with them.",
        "",
    ]
    for row in rows:
        chain = " -> ".join(reversed(row.get("dependency_chain") or []))
        lines.append(
            f"- product={row['product']} | component={row['component']} | "
            f"direct_supplier={row['direct_supplier']} | "
            f"tier_depth={row['tier_depth']} | sole_source={row['sole_source']}"
            + (f" | chain={chain}" if row["tier_depth"] else "")
        )
    return "\n".join(lines)


def _render_product_tree(product: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"Supply tree for product '{product}' (from the bill of materials "
             f"and the supplier graph).", ""]
    for row in rows:
        upstream = ", ".join(row.get("upstream_suppliers") or []) or "none mapped"
        locations = ", ".join(row.get("supplier_locations") or []) or "unknown"
        lines.append(
            f"- component={row['component']} (qty {row['quantity']}) | "
            f"supplier={row['supplier']} | sole_source={row['sole_source']} | "
            f"supplier_location={locations} | upstream={upstream}"
        )
    return "\n".join(lines)


def _render_component_chain(component: str, rows: list[dict[str, Any]],
                            shared: list[dict[str, Any]]) -> str:
    lines = [f"Supply chain for component '{component}'.", ""]
    for row in rows:
        upstream = ", ".join(row.get("upstream_suppliers") or []) or "none mapped"
        lines.append(
            f"- supplier={row['supplier']} | sole_source={row['sole_source']} | "
            f"share_pct={row['share_pct']} | "
            f"location={', '.join(row.get('supplier_locations') or []) or 'unknown'} | "
            f"upstream={upstream} | "
            f"upstream_location={', '.join(row.get('upstream_locations') or []) or 'unknown'}"
        )
        if row.get("used_in_products"):
            lines.append(f"  used in: {', '.join(row['used_in_products'])}")
    if shared:
        lines.append("")
        lines.append(
            "SHARED UPSTREAM TEST: the following upstream suppliers are reached "
            "from MORE THAN ONE of the suppliers above, meaning those suppliers "
            "are not independent of each other:"
        )
        for row in shared:
            lines.append(
                f"- {row['shared_upstream']} is reached through "
                f"{', '.join(row['reached_through'])} "
                f"(located: {', '.join(row.get('upstream_locations') or []) or 'unknown'})"
            )
    else:
        lines.append("")
        lines.append(
            "SHARED UPSTREAM TEST: no upstream supplier is reached from more "
            "than one supplier of this component, within the mapped depth."
        )
    return "\n".join(lines)
