"""Loading the structured backbone: products, components, BOM, suppliers, sites.

This module exists to make a point that the rest of the project depends on.

A lot of GraphRAG material implies that the whole graph should be extracted
from text by a language model.  That is a mistake wherever a system of record
already holds the fact.  Northwind's bill of materials lives in a PLM system.
Its approved vendor list lives in an ERP.  Those facts are already structured,
already correct, already governed, and already someone's job to maintain.
Running them through an LLM extractor would take data with 100% accuracy and
give it back at 95%, at a cost, with no benefit whatsoever.

So the split in this project is:

  Structured source (CSV here, an ERP export in reality)
      Products, components, bill of materials, tier-1 supply, sites, locations.
      Loaded verbatim.  provenance = 'erp'.  confidence = 1.0.

  Unstructured source (documents, via ingest/extract.py)
      Tier-2 and tier-3 dependencies, incidents, audit findings, regulations,
      certifications.  provenance = 'llm'.  confidence from the extractor.

The graph holds both, and every relationship says which it is.  That single
property is what lets a user - and the answer layer - treat "Meridian supplies
PCB-A7" and "Meridian depends on Formosa Substrate" as the different kinds of
claim that they are.  The first is a fact from a governed system.  The second
is a model's reading of a sentence, with the sentence attached.

Deciding which facts belong on which side of that line is the central design
judgement in a real GraphRAG system, and it is made here, in code, rather than
left implicit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..graph.client import GraphClient
from .resolve import EntityResolver, ResolvedEntity, make_key

# ---------------------------------------------------------------------------
# Writes.  Each is UNWIND-shaped for batching, and each sets provenance='erp'
# so the property is never absent rather than being defaulted at read time in
# five different places.
# ---------------------------------------------------------------------------
_CONTAINS = """
UNWIND $rows AS row
MATCH (p:Entity {key: row.product_key})
MATCH (c:Entity {key: row.component_key})
MERGE (p)-[r:CONTAINS]->(c)
SET r.quantity = row.quantity, r.provenance = 'erp', r.confidence = 1.0
"""

_SUPPLIES = """
UNWIND $rows AS row
MATCH (s:Entity {key: row.supplier_key})
MATCH (c:Entity {key: row.component_key})
MERGE (s)-[r:SUPPLIES]->(c)
SET r.sole_source = row.sole_source,
    r.share_pct = row.share_pct,
    r.qualified_since = row.qualified_since,
    r.provenance = 'erp',
    r.confidence = 1.0
"""

_OPERATES = """
UNWIND $rows AS row
MATCH (s:Entity {key: row.supplier_key})
MATCH (t:Entity {key: row.site_key})
MERGE (s)-[r:OPERATES]->(t)
SET r.provenance = 'erp', r.confidence = 1.0
"""

_LOCATED_IN = """
UNWIND $rows AS row
MATCH (t:Entity {key: row.site_key})
MATCH (l:Entity {key: row.location_key})
MERGE (t)-[r:LOCATED_IN]->(l)
SET r.provenance = 'erp', r.confidence = 1.0
"""


@dataclass
class BackboneStats:
    products: int = 0
    components: int = 0
    suppliers: int = 0
    sites: int = 0
    locations: int = 0
    contains: int = 0
    supplies: int = 0
    operates: int = 0
    located_in: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing structured input {path.name}. The backbone CSVs in "
            "data/structured/ are required - they stand in for the ERP and "
            "PLM exports a real deployment would pull."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _split_aliases(value: str) -> list[str]:
    return [a.strip() for a in (value or "").split("|") if a.strip()]


def load_backbone(client: GraphClient, resolver: EntityResolver,
                  directory: Path) -> BackboneStats:
    """Register backbone entities with the resolver AND write them to Neo4j.

    Registering before any document is extracted is what makes resolution work:
    the ERP spelling becomes the canonical name, and every later mention in
    prose resolves onto it rather than founding a competing node.
    """
    stats = BackboneStats()
    entity_rows: dict[str, list[dict[str, Any]]] = {}

    def add(entity: ResolvedEntity) -> None:
        entity_rows.setdefault(entity.type, []).append(
            {
                "key": entity.key,
                "name": entity.name,
                "aliases": entity.aliases,
                "type": entity.type,
                "summary": entity.summary,
                "status": entity.status,
            }
        )

    # -- products ----------------------------------------------------------
    for row in _read(directory / "products.csv"):
        add(resolver.register(
            "Product", row["name"],
            # "NW-500" must resolve to "NW-500 Patient Monitor". Prose uses the
            # short code far more often than the catalogue name.
            aliases=[row["name"].split()[0]],
            summary=row["description"], authoritative=True,
        ))
        stats.products += 1

    # -- components --------------------------------------------------------
    for row in _read(directory / "components.csv"):
        add(resolver.register(
            "Component", row["name"],
            # The bare part number is registered as an alias so that a document
            # saying "PCB-A7" resolves onto the same node as the full name
            # "PCB-A7 Main Controller Board" from the PLM.  Without this the
            # corpus and the ERP would build two disjoint graphs that never
            # touch, and every cross-source query would return nothing.
            aliases=[row["name"].split()[0]],
            summary=f"{row['category']}, lead time {row['lead_time_days']} days",
            authoritative=True,
        ))
        stats.components += 1

    # -- suppliers ---------------------------------------------------------
    for row in _read(directory / "suppliers.csv"):
        add(resolver.register(
            "Supplier", row["name"],
            aliases=_split_aliases(row["aliases"]),
            summary=f"Tier {row['tier']} supplier, {row['country']}, "
                    f"qualified since {row['relationship_since']}",
            authoritative=True,
        ))
        stats.suppliers += 1

    # -- sites and locations ----------------------------------------------
    site_rows = _read(directory / "sites.csv")
    seen_locations: set[str] = set()
    for row in site_rows:
        add(resolver.register("Site", row["name"], authoritative=True,
                              summary=f"{row['location_name']}, {row['country']}"))
        stats.sites += 1
        if row["location_name"] not in seen_locations:
            seen_locations.add(row["location_name"])
            add(resolver.register("Location", row["location_name"],
                                  summary=row["country"], authoritative=True))
            stats.locations += 1

    # -- write entities ----------------------------------------------------
    from ..graph.queries import upsert_entities  # local import: avoids a cycle
    for label, rows in entity_rows.items():
        client.run_batch(upsert_entities(label), rows)

    # -- write relationships ----------------------------------------------
    supplier_by_id = {r["supplier_id"]: r["name"] for r in _read(directory / "suppliers.csv")}
    component_by_id = {r["component_id"]: r["name"] for r in _read(directory / "components.csv")}
    product_by_id = {r["product_id"]: r["name"] for r in _read(directory / "products.csv")}

    contains = [
        {
            "product_key": make_key("Product", product_by_id[row["product_id"]]),
            "component_key": make_key("Component", component_by_id[row["component_id"]]),
            "quantity": int(row["quantity"]),
        }
        for row in _read(directory / "bom.csv")
    ]
    stats.contains = client.run_batch(_CONTAINS, contains)

    supplies = [
        {
            "supplier_key": make_key("Supplier", supplier_by_id[row["supplier_id"]]),
            "component_key": make_key("Component", component_by_id[row["component_id"]]),
            # CSV gives strings; Neo4j must receive a real boolean or the
            # `WHERE s.sole_source = true` filter in queries.py matches nothing
            # and reports "no sole-sourced components" on a base full of them.
            "sole_source": row["sole_source"].strip().lower() == "true",
            "share_pct": int(row["share_pct"]),
            "qualified_since": row["qualified_since"],
        }
        for row in _read(directory / "supplies.csv")
    ]
    stats.supplies = client.run_batch(_SUPPLIES, supplies)

    operates = [
        {
            "supplier_key": make_key("Supplier", supplier_by_id[row["supplier_id"]]),
            "site_key": make_key("Site", row["name"]),
        }
        for row in site_rows
    ]
    stats.operates = client.run_batch(_OPERATES, operates)

    located = [
        {
            "site_key": make_key("Site", row["name"]),
            "location_key": make_key("Location", row["location_name"]),
        }
        for row in site_rows
    ]
    stats.located_in = client.run_batch(_LOCATED_IN, located)

    return stats
