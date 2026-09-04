"""Graph exploration service.

Everything the explorer and the Cypher tab need, expressed as domain
operations rather than as HTTP handlers. See `src/services/qa.py` for the
layering rationale.

Nothing here imports fastapi. `run_cypher` raises ValueError for a rejected
query and lets the routing layer decide that ValueError means 400 - the service
does not know what a status code is.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..config import Config
from ..graph import queries
from ..graph.client import GraphClient
from ..graph.schema import index_status


# Write operations, rejected on the ad-hoc query endpoint.
#
# This is NOT a security boundary: anyone who can reach the API can reach the
# database. It guards against a learner destroying their own graph with a stray
# DETACH DELETE mid-lesson. The real boundary is Neo4j RBAC with a read-only
# role, which docs/security.md describes for production.
_WRITE_PATTERN = re.compile(
    r"\b(create|delete|detach|merge|set|remove|drop|load\s+csv|foreach|"
    r"call\s+db\.(create|index)|call\s+apoc\.(create|refactor|merge|periodic))\b",
    re.IGNORECASE,
)

# The Cypher cookbook. Served from the backend so the UI never hardcodes a
# query - what a learner reads in the app is what actually runs.
COOKBOOK: dict[str, tuple[str, dict, str]] = {
    "Products exposed to a disruption at a location": (
        queries.PRODUCTS_EXPOSED_TO_LOCATION,
        {"location_key": "location:kaohsiung", "limit": 25},
        "The flagship multi-hop query. Walks Location <- Site <- Supplier "
        "<- DEPENDS_ON* -> Supplier -> Component <- Product: five relationship "
        "types in one statement. The answer exists in no document in the "
        "corpus - the incident bulletin explicitly declines to give it.",
    ),
    "Downstream impact of one supplier": (
        queries.SUPPLIER_DOWNSTREAM_IMPACT,
        {"key": "supplier:sarawak-copper-foil", "limit": 25},
        "The same traversal seeded on a supplier instead of a place. Sarawak is "
        "a tier-3 copper foil producer appearing in exactly one document, which "
        "names only its immediate customer. Getting from there to a finished "
        "product is three further hops through two other documents and a CSV.",
    ),
    "Shared upstream behind a dual-sourced component": (
        queries.SHARED_UPSTREAM_FOR_COMPONENT,
        {"key": "part:li-18650-battery-pack"},
        "The set intersection that decides whether dual sourcing is real. Which "
        "suppliers supply the part is an ERP fact; what each buys upstream is "
        "in two separate documents. No single record performs the join.",
    ),
    "Sole-sourced parts whose supplier has an OPEN finding": (
        queries.SOLE_SOURCE_WITH_FINDINGS, {},
        "Joins a structured attribute (sole_source, from the ERP) against an "
        "LLM-extracted one (an open audit finding, from a PDF). Neither source "
        "answers it alone. Note `status = 'open'`: without it the query also "
        "returns every supplier whose only finding was a minor observation "
        "closed during the audit, which is most of them - and a risk report "
        "that flags everybody flags nobody.",
    ),
    "Supplier criticality by product fan-out": (
        queries.SUPPLIER_CRITICALITY, {"limit": 15},
        "How many finished products would eventually be affected if this "
        "supplier stopped shipping, at any tier. Pure structure - no text, no "
        "model. Note the tier-2 and tier-3 names near the top: the company has "
        "no contract with them and no purchase order names them.",
    ),
    "Full supply tree for one product": (
        queries.PRODUCT_SUPPLY_TREE,
        {"key": "product:nw-500-patient-monitor", "limit": 25},
        "Everything beneath one finished product, to the depth we have mapped.",
    ),
}


class GraphService:
    def __init__(self, *, graph: GraphClient, config: Config) -> None:
        self.graph = graph
        self.config = config

    # ---------------------------------------------------------------- census
    def census(self) -> dict[str, Any]:
        counts = self.graph.counts()
        nodes = {k.split(":", 1)[1]: v for k, v in counts.items()
                 if k.startswith("node:")}
        rels = {k.split(":", 1)[1]: v for k, v in counts.items()
                if k.startswith("rel:")}
        return {
            "nodes": nodes, "relationships": rels,
            "total_nodes": sum(nodes.values()),
            "total_relationships": sum(rels.values()),
            "populated": bool(nodes),
        }

    def schema(self) -> dict[str, Any]:
        """Indexes and constraints as the DATABASE reports them, not as the code
        claims them. The difference matters: a vector index that failed to come
        online returns no rows and no error."""
        return {
            "indexes": index_status(self.graph),
            "knowledge_relationships": queries.KNOWLEDGE_RELS,
            "entity_types": self.config.extraction["entity_types"],
            "embedding_dimensions": self.config.embedding["dimensions"],
            "max_traversal_hops": queries.MAX_ALLOWED_HOPS,
        }

    # -------------------------------------------------------------- entities
    def entities(self, *, entity_type: str | None = None,
                 search: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if entity_type:
            clauses.append("e.type = $entity_type")
            params["entity_type"] = entity_type
        if search:
            clauses.append("toLower(e.name) CONTAINS toLower($search)")
            params["search"] = search
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.graph.run(
            f"MATCH (e:Entity) {where} RETURN e.key AS key, e.name AS name, "
            "e.type AS type ORDER BY e.type, e.name LIMIT $limit",
            **params,
        )

    def entity_detail(self, key: str) -> dict[str, Any] | None:
        rows = self.graph.run(queries.ENTITY_DETAIL, key=key)
        if not rows:
            return None
        entity = rows[0]

        # Provenance for every edge. This is the point of the explorer: a user
        # must never have to guess which relationships are model inferences.
        evidence: list[dict[str, Any]] = []
        for neighbour in entity.get("neighbours") or []:
            if not neighbour.get("name"):
                continue
            from_key = key if neighbour["direction"] == "out" else neighbour["key"]
            to_key = neighbour["key"] if neighbour["direction"] == "out" else key
            for row in self.graph.run(queries.RELATIONSHIP_EVIDENCE,
                                      from_key=from_key, to_key=to_key):
                evidence.append({"neighbour": neighbour["name"],
                                 "direction": neighbour["direction"], **row})

        return {
            "key": entity["key"], "name": entity["name"], "type": entity["type"],
            "aliases": entity.get("aliases") or [],
            "summary": entity.get("summary"),
            "neighbours": [n for n in (entity.get("neighbours") or []) if n.get("name")],
            "documents": [d for d in (entity.get("documents") or []) if d.get("doc_id")],
            "evidence": evidence,
        }

    def subgraph(self, keys: list[str], *, hops: int = 2,
                 limit: int = 60) -> dict[str, Any]:
        nodes = self.graph.run(queries.subgraph_nodes(hops), keys=keys, limit=limit)
        node_keys = [n["key"] for n in nodes]
        # Edges filtered to the nodes actually returned, so the drawing can never
        # contain an edge dangling to a node the LIMIT cut.
        edges = self.graph.run(queries.SUBGRAPH_EDGES, keys=node_keys,
                               rel_types=queries.KNOWLEDGE_RELS)
        return {"nodes": nodes, "edges": edges, "seeds": keys}

    # -------------------------------------------------------------- cookbook
    @staticmethod
    def cookbook() -> list[dict[str, Any]]:
        return [
            {"name": name, "cypher": cypher.strip(), "parameters": params,
             "explanation": explanation}
            for name, (cypher, params, explanation) in COOKBOOK.items()
        ]

    def run_cypher(self, cypher: str,
                   parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        if _WRITE_PATTERN.search(cypher):
            raise ValueError(
                "Write operations are blocked on this endpoint. It exists so "
                "you can explore the graph without being able to destroy it by "
                "accident. Use `python run.py ingest` to change the graph."
            )
        started = time.perf_counter()
        rows = self.graph.run(cypher, **(parameters or {}))
        return {
            "rows": rows[:500], "row_count": len(rows),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
