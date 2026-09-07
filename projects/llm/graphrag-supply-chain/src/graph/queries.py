"""Every Cypher query the application runs, in one file, with its reasoning.

This module is also the project's Cypher teaching material.  Each query is a
module-level constant with a comment explaining what it does, why it is shaped
that way, and what the naive version would have cost.  The Streamlit app shows
learners the exact query it just executed, pulled from these constants - so
what you read here is what actually runs, not a prettified copy.

A note on parameters.  Every value arrives as ``$param``, never as an f-string.
Two reasons, and the second is the one people forget:

  1. Cypher injection is real.  ``"MATCH (n {name:'" + user_input + "'})"`` is
     the same bug as SQL injection with the same consequences.
  2. Neo4j caches an execution plan per query *string*.  Interpolating values
     produces a new string every call, so the plan cache misses every time and
     you pay the planner on every request.  Parameters keep one plan hot.

The single exception is traversal depth, because Cypher does not allow
``*1..$hops``.  ``_depth`` below validates it as a small integer before it can
reach a query string.
"""

from __future__ import annotations

# Relationship types that make up the knowledge subgraph.  Traversal is
# restricted to these so an expansion can never wander into the text subgraph
# (:MENTIONS, :PART_OF) and come back with "these two suppliers are related
# because the same PDF mentioned both".  That is not a relationship, it is a
# coincidence, and letting it into a path is how a GraphRAG system starts
# producing confident nonsense.
KNOWLEDGE_RELS = [
    "DEPENDS_ON", "OPERATES", "LOCATED_IN", "SUPPLIES",
    "CONTAINS", "AFFECTS", "RAISED_AGAINST", "HOLDS", "APPLIES_TO",
]
_REL_PATTERN = "|".join(KNOWLEDGE_RELS)

MAX_ALLOWED_HOPS = 5


def _depth(hops: int) -> int:
    """Validate traversal depth before it is interpolated into Cypher.

    Depth cannot be a query parameter, so it is the one value that gets
    formatted into the string - which means it is the one value that must be
    proven safe first.  It is also a cost control: an unbounded ``*`` on a
    dense graph is how you take a Neo4j instance down.
    """
    hops = int(hops)
    if not 1 <= hops <= MAX_ALLOWED_HOPS:
        raise ValueError(
            f"traversal depth must be between 1 and {MAX_ALLOWED_HOPS}, got {hops}"
        )
    return hops


# ===========================================================================
# 1. VECTOR RETRIEVAL  -  the baseline every RAG system has
# ===========================================================================
# db.index.vector.queryNodes is an approximate nearest-neighbour search over
# the HNSW index built in schema.py.  It returns nodes with a cosine score in
# [0,1].  Note we return the chunk's document too: a citation that says
# "chunk 47" is useless to a human, and groundedness you cannot check is not
# groundedness.
VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index_name, $k, $embedding)
YIELD node AS chunk, score
MATCH (chunk)-[:PART_OF]->(doc:Document)
RETURN chunk.chunk_id  AS chunk_id,
       chunk.text      AS text,
       doc.doc_id      AS doc_id,
       doc.title       AS title,
       doc.doc_type    AS doc_type,
       score           AS score
ORDER BY score DESC
"""

# ===========================================================================
# 2. KEYWORD RETRIEVAL  -  BM25, for the terms embeddings are bad at
# ===========================================================================
# Part numbers ("PCB-A7"), standard names ("IATF 16949") and codes carry almost
# no distributional meaning, so they sit in dense space next to every other
# alphanumeric token.  Lucene matches them exactly.  This is why the honest
# baseline in the comparison tab is vector+BM25, not vector alone.
FULLTEXT_CHUNK_SEARCH = """
CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $k})
YIELD node AS chunk, score
MATCH (chunk)-[:PART_OF]->(doc:Document)
RETURN chunk.chunk_id  AS chunk_id,
       chunk.text      AS text,
       doc.doc_id      AS doc_id,
       doc.title       AS title,
       doc.doc_type    AS doc_type,
       score           AS score
ORDER BY score DESC
"""

# ===========================================================================
# 3. ENTITY LINKING  -  question text -> nodes in the graph
# ===========================================================================
# The entry point to every graph retrieval.  Full-text over name + aliases,
# which is why entity resolution writes aliases onto the surviving node instead
# of discarding them: "Formosa Substrate", "Formosa Substrate Materials Co."
# and "FSM" must all land on the same node.
ENTITY_LINK = """
CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $k})
YIELD node AS entity, score
RETURN entity.key    AS key,
       entity.name   AS name,
       entity.type   AS type,
       score         AS score
ORDER BY score DESC
"""

ENTITIES_BY_KEY = """
MATCH (e:Entity) WHERE e.key IN $keys
RETURN e.key AS key, e.name AS name, e.type AS type,
       e.aliases AS aliases, e.summary AS summary
"""

# ===========================================================================
# 4. THE BRIDGE  -  chunk -> entities, and entities -> chunks
# ===========================================================================
# These two queries are the hinge of the whole architecture.  The first turns a
# vector hit into graph anchors; the second turns traversal results back into
# quotable evidence.  Everything else is plumbing around them.
CHUNK_ENTITIES = """
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE c.chunk_id IN $chunk_ids AND m.confidence >= $min_confidence
RETURN DISTINCT e.key AS key, e.name AS name, e.type AS type,
       max(m.confidence) AS confidence
ORDER BY confidence DESC
"""

# Evidence for a set of entities.  ``mention_count`` ranks chunks that discuss
# several of our entities above chunks that merely name one in passing - a
# cheap, deterministic relevance signal that costs no model call.
CHUNKS_FOR_ENTITIES = """
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE e.key IN $keys AND m.confidence >= $min_confidence
WITH c, count(DISTINCT e) AS mention_count, collect(DISTINCT e.name) AS entities
MATCH (c)-[:PART_OF]->(doc:Document)
RETURN c.chunk_id AS chunk_id,
       c.text     AS text,
       doc.doc_id AS doc_id,
       doc.title  AS title,
       doc.doc_type AS doc_type,
       mention_count,
       entities
ORDER BY mention_count DESC, c.chunk_id
LIMIT $limit
"""


# ===========================================================================
# 5. NEIGHBOURHOOD EXPANSION  -  "what is near these entities?"
# ===========================================================================
def neighbourhood(hops: int) -> str:
    """Undirected variable-length expansion from a set of seed entities.

    Undirected on purpose.  ``(a)-[:SUPPLIES]->(b)`` and its reverse are the
    same fact seen from two sides, and a risk question does not care which way
    the arrow was drawn when the data was modelled.  Directed traversal here
    would silently miss half the exposure.

    The ``LIMIT`` is not decoration.  Undirected expansion at depth 3 through a
    hub node (a location every supplier sits in) is combinatorial; without a
    cap this query is a denial-of-service against your own database.
    """
    return f"""
    MATCH (seed:Entity) WHERE seed.key IN $keys
    MATCH path = (seed)-[r:{_REL_PATTERN}*1..{_depth(hops)}]-(other:Entity)
    WHERE ALL(rel IN r WHERE coalesce(rel.confidence, 1.0) >= $min_confidence)
    WITH other, path, length(path) AS distance
    ORDER BY distance ASC
    WITH other, min(distance) AS hops, head(collect(path)) AS shortest
    RETURN other.key  AS key,
           other.name AS name,
           other.type AS type,
           hops       AS hops,
           [n IN nodes(shortest) | n.name] AS path_names,
           [rel IN relationships(shortest) | type(rel)] AS path_rels
    ORDER BY hops ASC, name ASC
    LIMIT $limit
    """


# ===========================================================================
# 6. THE FLAGSHIP QUERY  -  exposure of finished products to a location
# ===========================================================================
# This is the query no vector search can imitate, and it is worth reading
# slowly.  It walks five relationship types in one statement:
#
#   Location <-LOCATED_IN- Site <-OPERATES- Supplier      (who is there)
#      <-DEPENDS_ON*0..3-  Supplier                       (who needs them)
#      -SUPPLIES-> Component <-CONTAINS- Product          (what we ship)
#
# ``*0..3`` includes zero hops, so a tier-1 supplier that is itself in the
# affected location is returned alongside the deep-tier cases.  Dropping the
# zero is a classic off-by-one that hides the most obvious exposure of all.
#
# The answer to "which products are exposed to a shutdown in Kaohsiung" exists
# in no document in the corpus.  It exists only here, as a join.  That is the
# entire argument for this project.
#
# One robustness note that is worth more than it looks.  The location leg is
# written as a pattern *predicate* over two shapes:
#
#     Supplier -[:OPERATES]-> Site -[:LOCATED_IN]-> Location    (the ERP shape)
#     Supplier -[:LOCATED_IN]-> Location                        (what the
#                                                                extractor
#                                                                often emits)
#
# The tier-1 backbone comes from CSV and always has a Site.  The tier-2
# suppliers come from prose, where a sentence like "Formosa operates a single
# line at a facility in Kaohsiung" may or may not cause the model to mint a
# Site node.  Both readings are correct; insisting on one of them would make
# the flagship query silently return half the exposure depending on how a
# model felt about a sentence.  Accepting both is not sloppiness, it is
# defending a query against upstream variability you do not control.
PRODUCTS_EXPOSED_TO_LOCATION = """
MATCH (loc:Location) WHERE loc.key = $location_key
MATCH (origin:Supplier)
WHERE (origin)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(loc)
   OR (origin)-[:LOCATED_IN]->(loc)
OPTIONAL MATCH (origin)-[:OPERATES]->(site:Site)-[:LOCATED_IN]->(loc)
MATCH depends = (dependent:Supplier)-[:DEPENDS_ON*0..3]->(origin)
MATCH (dependent)-[s:SUPPLIES]->(comp:Component)<-[:CONTAINS]-(prod:Product)
RETURN DISTINCT
       prod.name       AS product,
       comp.name       AS component,
       dependent.name  AS direct_supplier,
       origin.name     AS exposed_supplier,
       coalesce(site.name, loc.name) AS exposed_site,
       length(depends) AS tier_depth,
       coalesce(s.sole_source, false) AS sole_source,
       [n IN nodes(depends) | n.name] AS dependency_chain
ORDER BY tier_depth ASC, product, component
LIMIT $limit
"""

# The same traversal seeded on a supplier instead of a location: "if this
# company stopped shipping, what eventually stops?"  Seeding on a tier-3
# supplier like a copper foil producer walks foil -> laminate -> boards ->
# products, which is four relationship hops and zero documents.
SUPPLIER_DOWNSTREAM_IMPACT = """
MATCH (origin:Supplier {key: $key})
MATCH depends = (dependent:Supplier)-[:DEPENDS_ON*0..3]->(origin)
MATCH (dependent)-[s:SUPPLIES]->(comp:Component)<-[:CONTAINS]-(prod:Product)
RETURN DISTINCT
       prod.name       AS product,
       comp.name       AS component,
       dependent.name  AS direct_supplier,
       origin.name     AS origin_supplier,
       length(depends) AS tier_depth,
       coalesce(s.sole_source, false) AS sole_source,
       [n IN nodes(depends) | n.name] AS dependency_chain
ORDER BY tier_depth ASC, product, component
LIMIT $limit
"""

# ===========================================================================
# 7. PATH QUERIES  -  "how are these two things connected?"
# ===========================================================================
# shortestPath is a built-in graph algorithm, not a loop you write.  Neo4j
# expands from both ends simultaneously and stops when the frontiers meet.
# The equivalent in a relational database is a recursive CTE that gets slower
# with every hop; here the cost is proportional to the size of the frontier,
# not the size of the table.  This asymmetry is the actual reason graph
# databases exist.
SHORTEST_PATH = """
MATCH (a:Entity {key: $from_key}), (b:Entity {key: $to_key})
MATCH path = shortestPath((a)-[:%s*1..%d]-(b))
RETURN [n IN nodes(path) | {key: n.key, name: n.name, type: n.type}] AS nodes,
       [r IN relationships(path) | type(r)] AS rels,
       length(path) AS hops
""" % (_REL_PATTERN, MAX_ALLOWED_HOPS)

# ===========================================================================
# 8. RISK QUERIES  -  what the business actually asks
# ===========================================================================
# Sole-sourced components whose supplier has an open audit finding.  Note this
# joins a structured fact (sole_source, from the ERP export) against an
# LLM-extracted fact (the finding, from an audit PDF).  Neither source can
# answer it alone, which is the practical argument for putting both in one
# graph instead of one in a warehouse and the other in a vector store.
SOLE_SOURCE_WITH_FINDINGS = """
MATCH (sup:Supplier)-[s:SUPPLIES]->(comp:Component)<-[:CONTAINS]-(prod:Product)
WHERE s.sole_source = true
// `status = 'open'` is load-bearing. Without it this query also returns every
// supplier whose only finding was a minor observation closed during the audit,
// which is most of them - and a risk report that flags everybody flags nobody.
OPTIONAL MATCH (f:Finding)-[:RAISED_AGAINST]->(sup)
WHERE f.status = 'open'
WITH sup, comp, prod, collect(DISTINCT f.name) AS findings
WHERE size(findings) > 0
RETURN sup.name AS supplier, comp.name AS component,
       collect(DISTINCT prod.name) AS products, findings
ORDER BY supplier
"""

# Single points of failure by fan-out: which suppliers, if lost, take down the
# most finished products.  Pure structure, no text, no model.
SUPPLIER_CRITICALITY = """
MATCH (sup:Supplier)-[:DEPENDS_ON*0..3]->(root:Supplier)
MATCH (sup)-[:SUPPLIES]->(:Component)<-[:CONTAINS]-(prod:Product)
WITH root, collect(DISTINCT prod.name) AS products
RETURN root.name AS supplier, size(products) AS products_at_risk, products
ORDER BY products_at_risk DESC, supplier
LIMIT $limit
"""

# ===========================================================================
# 9. VISUALISATION  -  the subgraph the UI draws
# ===========================================================================
# Two queries rather than one clever one.  A single statement returning nodes
# and edges together needs nested aggregation that is hard to read and easy to
# get subtly wrong, and the drawing library wants two lists anyway.  Two simple
# round trips beat one unreadable one.
SUBGRAPH_NODES = """
MATCH (seed:Entity) WHERE seed.key IN $keys
MATCH (seed)-[:%s*0..%%d]-(other:Entity)
RETURN DISTINCT other.key AS key, other.name AS name, other.type AS type,
       other.key IN $keys AS is_seed
LIMIT $limit
""" % _REL_PATTERN

# Edges are then filtered to the node set we actually drew, so the picture can
# never contain a dangling edge to a node that was cut by the LIMIT.
SUBGRAPH_EDGES = """
MATCH (a:Entity)-[r]->(b:Entity)
WHERE a.key IN $keys AND b.key IN $keys AND type(r) IN $rel_types
RETURN startNode(r).key AS source, endNode(r).key AS target,
       type(r) AS type, coalesce(r.provenance, 'erp') AS provenance,
       coalesce(r.confidence, 1.0) AS confidence
"""


def subgraph_nodes(hops: int) -> str:
    """Seed set plus everything within ``hops``.  ``*0..`` keeps the seeds
    themselves in the result even when they are isolated - otherwise clicking
    an entity with no neighbours draws an empty canvas and looks broken."""
    return SUBGRAPH_NODES % _depth(hops)

# Everything attached to one entity - the "inspect a node" panel.
ENTITY_DETAIL = """
MATCH (e:Entity {key: $key})
OPTIONAL MATCH (e)-[r]-(other:Entity)
WITH e, collect(DISTINCT {
       name: other.name, key: other.key, type: other.type,
       rel: type(r), direction: CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END
     }) AS neighbours
OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
RETURN e.key AS key, e.name AS name, e.type AS type,
       e.aliases AS aliases, e.summary AS summary,
       neighbours,
       collect(DISTINCT {doc_id: d.doc_id, title: d.title})[0..10] AS documents
"""

# ===========================================================================
# 10. INGESTION WRITES
# ===========================================================================
# All UNWIND-shaped; see GraphClient.run_batch for why that matters.
#
# ON CREATE / ON MATCH on the entity MERGE is the entity-resolution mechanism
# at the database level: a second document mentioning the same supplier under a
# different surface form adds its alias to the existing node rather than
# creating a rival one.  ``apoc.coll.toSet`` would be tidier; this project
# stays APOC-free so it runs on Aura Free, so the de-duplication is done with
# plain list comprehension.
UPSERT_DOCUMENTS = """
UNWIND $rows AS row
MERGE (d:Document {doc_id: row.doc_id})
SET d.title = row.title,
    d.doc_type = row.doc_type,
    d.source_path = row.source_path,
    d.published = row.published
"""

UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (d:Document {doc_id: row.doc_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.text = row.text,
    c.ord = row.ord,
    c.doc_id = row.doc_id,
    c.embedding = row.embedding
MERGE (c)-[:PART_OF]->(d)
"""

UPSERT_MENTIONS = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e:Entity {key: row.key})
MERGE (c)-[m:MENTIONS]->(e)
SET m.confidence = row.confidence
"""


def upsert_entities(label: str) -> str:
    """MERGE entities of one type.

    Cypher cannot parameterise a label, so this is generated per type - and the
    caller must pass a label from ``extraction.entity_types``, never a string
    that came from a model.  ``ingest/pipeline.py`` enforces that.

    The double label (:Entity:{label}) is the decision documented at the top of
    schema.py.  MERGE keys on :Entity only, so the uniqueness constraint
    applies across all types.
    """
    return f"""
    UNWIND $rows AS row
    MERGE (e:Entity {{key: row.key}})
    ON CREATE SET e.name = row.name, e.aliases = row.aliases,
                  e.type = row.type, e.summary = row.summary,
                  e.status = row.status, e.created_at = timestamp()
    ON MATCH  SET e.aliases = e.aliases +
                              [a IN row.aliases WHERE NOT a IN e.aliases],
                  e.summary = coalesce(e.summary, row.summary),
                  e.status = coalesce(e.status, row.status)
    SET e:{label}
    """


UPSERT_RELATIONSHIPS_TEMPLATE = """
UNWIND $rows AS row
MATCH (a:Entity {key: row.source})
MATCH (b:Entity {key: row.target})
MERGE (a)-[r:%s]->(b)
SET r.confidence = row.confidence,
    r.source_doc = row.source_doc,
    r.evidence = row.evidence,
    r.provenance = row.provenance
"""


def upsert_relationships(rel_type: str) -> str:
    """Same label-parameterisation problem as entities, same defence: the
    caller may only pass a type from ``extraction.relation_types``.

    Every relationship carries ``provenance`` ('erp' or 'llm'), ``source_doc``
    and a verbatim ``evidence`` span.  That is not bookkeeping - it is what
    lets the answer layer cite a traversal step, and what lets a human audit an
    LLM-invented edge.  An edge you cannot trace back to a sentence is an edge
    you cannot defend.
    """
    return UPSERT_RELATIONSHIPS_TEMPLATE % rel_type


# ===========================================================================
# 11. TYPE-DRIVEN TEMPLATES
# ===========================================================================
# These back the deterministic query planner in retrieval/strategies.py.  Which
# template fires is decided by the *type* of entity linked from the question
# (a Location, a Product, a Component), never by keyword-matching the question
# text.  Type-driven routing is stable; keyword routing breaks the first time
# someone phrases a question differently, and breaks silently.

# Everything beneath one finished product, to the depth we have mapped.
PRODUCT_SUPPLY_TREE = """
MATCH (prod:Product {key: $key})-[cont:CONTAINS]->(comp:Component)
OPTIONAL MATCH (sup:Supplier)-[s:SUPPLIES]->(comp)
OPTIONAL MATCH (sup)-[:DEPENDS_ON*1..3]->(upstream:Supplier)
OPTIONAL MATCH (sup)-[:OPERATES]->(site:Site)-[:LOCATED_IN]->(loc:Location)
RETURN comp.name AS component,
       cont.quantity AS quantity,
       sup.name AS supplier,
       coalesce(s.sole_source, false) AS sole_source,
       collect(DISTINCT upstream.name) AS upstream_suppliers,
       collect(DISTINCT loc.name) AS supplier_locations
ORDER BY component
LIMIT $limit
"""

# Everything above one component: who supplies it, who they depend on, where
# those sit, and which products consume it.
COMPONENT_SUPPLY_CHAIN = """
MATCH (comp:Component {key: $key})
OPTIONAL MATCH (sup:Supplier)-[s:SUPPLIES]->(comp)
OPTIONAL MATCH (sup)-[:DEPENDS_ON*1..3]->(upstream:Supplier)
OPTIONAL MATCH (upstream)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(uloc:Location)
OPTIONAL MATCH (sup)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(sloc:Location)
OPTIONAL MATCH (prod:Product)-[:CONTAINS]->(comp)
RETURN comp.name AS component,
       collect(DISTINCT prod.name) AS used_in_products,
       sup.name AS supplier,
       coalesce(s.sole_source, false) AS sole_source,
       coalesce(s.share_pct, 0) AS share_pct,
       collect(DISTINCT sloc.name) AS supplier_locations,
       collect(DISTINCT upstream.name) AS upstream_suppliers,
       collect(DISTINCT uloc.name) AS upstream_locations
ORDER BY supplier
LIMIT $limit
"""

# The shared-upstream test: for one component, do its nominally independent
# suppliers converge on the same upstream supplier?  This is the query that
# answers "is our dual sourcing real", and it is a pure set intersection that
# no document in the corpus performs.
SHARED_UPSTREAM_FOR_COMPONENT = """
MATCH (sup:Supplier)-[:SUPPLIES]->(comp:Component {key: $key})
MATCH (sup)-[:DEPENDS_ON*1..3]->(upstream:Supplier)
WITH upstream, collect(DISTINCT sup.name) AS via, comp
WHERE size(via) > 1
OPTIONAL MATCH (upstream)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(loc:Location)
RETURN comp.name AS component,
       upstream.name AS shared_upstream,
       via AS reached_through,
       collect(DISTINCT loc.name) AS upstream_locations
ORDER BY size(via) DESC
"""

# Evidence behind an LLM-extracted edge: the sentence, and the document.
RELATIONSHIP_EVIDENCE = """
MATCH (a:Entity {key: $from_key})-[r]->(b:Entity {key: $to_key})
RETURN type(r) AS type,
       coalesce(r.provenance, 'erp') AS provenance,
       coalesce(r.confidence, 1.0) AS confidence,
       r.evidence AS evidence,
       r.source_doc AS source_doc
"""
