"""The Neo4j schema: labels, constraints, and the three indexes.

Read this file before any other.  It is the data model, and in a graph project
the data model *is* the architecture.  Every design choice below is commented
with why it was made and what the alternative would have cost.

---------------------------------------------------------------------------
THE MODEL
---------------------------------------------------------------------------

Two subgraphs that share one bridge.

  A. The TEXT subgraph  (what RAG normally has)

        (:Document)<-[:PART_OF]-(:Chunk {text, embedding})

  B. The KNOWLEDGE subgraph  (what makes this GraphRAG)

        (:Product)-[:CONTAINS]->(:Component)<-[:SUPPLIES]-(:Supplier)
        (:Supplier)-[:DEPENDS_ON]->(:Supplier)        <- the tier-2/3 gold
        (:Supplier)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(:Location)
        (:Incident)-[:AFFECTS]->(:Site | :Supplier | :Location)
        (:Finding)-[:RAISED_AGAINST]->(:Supplier)
        (:Regulation)-[:APPLIES_TO]->(:Component | :Material)
        (:Supplier)-[:HOLDS]->(:Certification)

  THE BRIDGE

        (:Chunk)-[:MENTIONS {confidence}]->(:Entity)

The bridge is the whole system.  Vector search lands you on a chunk; MENTIONS
turns that chunk into entities; the knowledge subgraph turns those entities
into *other* entities several hops away; MENTIONS run backwards turns those
into chunks the vector search never would have scored.  That round trip -
text -> structure -> text - is the mechanism by which GraphRAG retrieves
evidence that is not similar to the question.

---------------------------------------------------------------------------
WHY EVERY ENTITY CARRIES TWO LABELS
---------------------------------------------------------------------------

A supplier node is created as ``(:Entity:Supplier)``.  Neo4j allows multiple
labels per node and here both earn their place:

  :Entity    lets one uniqueness constraint, one full-text index and one
             generic traversal cover all ten entity types.  Without it you
             need ten constraints and a ten-branch UNION for "find anything
             called X".
  :Supplier  lets you write ``MATCH (s:Supplier)`` and have the planner scan
             only suppliers instead of filtering every entity by a property.
             A label is an index; ``WHERE n.type = 'Supplier'`` is not.

The cost is one extra label per node and the discipline of always writing both
at creation time.  ``upsert_entities`` below is the only place that happens.
"""

from __future__ import annotations

from .client import GraphClient

# ---------------------------------------------------------------------------
# CONSTRAINTS
#
# A uniqueness constraint in Neo4j does two things at once: it rejects
# duplicates AND it creates a backing index.  So these are not merely
# integrity rules, they are the lookup path for every MERGE in the ingestion
# pipeline.  Without them, MERGE on a 5,000-node graph does a full label scan
# each time and ingestion goes quadratic.
#
# IF NOT EXISTS makes the whole file idempotent: running it twice is a no-op,
# which is what you want from a migration you will run on every startup.
# ---------------------------------------------------------------------------
CONSTRAINTS = [
    # `key` is the normalised identity of an entity (see ingest/resolve.py).
    # This constraint is what makes entity resolution *enforceable* rather than
    # merely attempted: two spellings that resolve to the same key cannot
    # become two nodes, because the database will not allow it.
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.key IS UNIQUE",

    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",

    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
]

# ---------------------------------------------------------------------------
# PLAIN INDEXES
#
# Range indexes on properties we filter by but do not MERGE on.  Cheap to
# maintain, and they turn the "show me all suppliers in Taiwan" panel in the
# UI from a scan into a seek.
# ---------------------------------------------------------------------------
INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
]

# ---------------------------------------------------------------------------
# VECTOR INDEX
#
# This is what makes Neo4j a vector store as well as a graph store, and it is
# the reason this project does not need a separate vector database.  One
# system holds the embeddings AND the relationships, so a retrieval can hop
# from a vector hit into a traversal without a network round trip or a join
# key that has to be kept in sync between two databases.
#
# Two settings, both load-bearing:
#
#   vector.dimensions          MUST equal configs/base.yaml -> embedding.dimensions.
#                              Mismatch is the single most common failure in
#                              this stack, and its symptom is not an error: the
#                              write is rejected or the query returns nothing,
#                              silently.  src/ingest/pipeline.py asserts the
#                              two agree before writing anything.
#
#   vector.similarity_function 'cosine' because Gemini embeddings are not
#                              normalised to unit length in a way that makes
#                              euclidean meaningful.  Cosine compares
#                              direction, which is what a semantic embedding
#                              encodes.
#
# Note this is created with a parameterised dimension, so changing the config
# and re-running `run.py ingest --reset` genuinely rebuilds it.
# ---------------------------------------------------------------------------
VECTOR_INDEX_NAME = "chunk_embedding_index"

VECTOR_INDEX_TEMPLATE = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dims},
  `vector.similarity_function`: 'cosine'
}}}}
"""

# ---------------------------------------------------------------------------
# FULL-TEXT INDEXES
#
# Two of them, for two different jobs, and the distinction matters.
#
#   entity_name_index   Entity linking.  A question says "Kaohsiung" and we
#                       need the Location node.  This is a *lexical* problem,
#                       not a semantic one - the user typed the name - and
#                       Lucene's analyser handles it better and ~200x cheaper
#                       than an embedding round trip.  Using vector similarity
#                       for name lookup is a real and common mistake: it
#                       cheerfully returns "Kaohsiung Precision Glass" when you
#                       asked about the city, because those two strings are
#                       semantically close.
#
#   chunk_text_index    BM25 keyword retrieval over chunk text.  We need it to
#                       build the *hybrid* baseline honestly: "GraphRAG beats
#                       vector search" is a much weaker claim if the vector
#                       baseline is missing exact-match terms that a 1970s
#                       keyword index would have caught.  Part numbers like
#                       "PCB-A7" are exactly that case - they carry almost no
#                       semantic signal and embeddings are bad at them.
# ---------------------------------------------------------------------------
ENTITY_FULLTEXT_INDEX = "entity_name_index"
CHUNK_FULLTEXT_INDEX = "chunk_text_index"

FULLTEXT_INDEXES = [
    f"CREATE FULLTEXT INDEX {ENTITY_FULLTEXT_INDEX} IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.aliases]",

    f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS "
    "FOR (c:Chunk) ON EACH [c.text]",
]


def apply_schema(client: GraphClient, dimensions: int) -> list[str]:
    """Create every constraint and index.  Idempotent - safe on every startup.

    Returns the statements executed, so the CLI and the notebook can print
    exactly what was run rather than claiming success abstractly.
    """
    statements = list(CONSTRAINTS) + list(INDEXES) + list(FULLTEXT_INDEXES)
    statements.append(
        VECTOR_INDEX_TEMPLATE.format(name=VECTOR_INDEX_NAME, dims=dimensions).strip()
    )
    for statement in statements:
        client.run_write(statement)

    # Index creation in Neo4j is asynchronous.  Querying a vector index that is
    # still POPULATING returns zero rows and no error, which during ingestion
    # looks exactly like "the embeddings did not save".  Waiting here removes a
    # whole class of phantom bug reports.
    client.run("CALL db.awaitIndexes(120000)")
    return statements


def index_status(client: GraphClient) -> list[dict]:
    """What Neo4j thinks exists.  Surfaced in the UI's Schema tab so a learner
    can see the constraints and indexes as real database objects rather than as
    strings in a Python file."""
    return client.run(
        "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, "
        "properties, state RETURN *"
    )


def verify_vector_index(client: GraphClient, dimensions: int) -> None:
    """Fail loudly if the index dimension and the configured dimension differ.

    This check exists because the failure it catches is silent.  Neo4j will not
    tell you that your 768-dimension query vector found nothing in a
    3072-dimension index; it returns an empty list, retrieval returns no
    context, the LLM says "I don't have enough information", and you spend an
    afternoon debugging your prompt.
    """
    rows = client.run(
        "SHOW INDEXES YIELD name, type, options "
        "WHERE name = $name RETURN options AS options",
        name=VECTOR_INDEX_NAME,
    )
    if not rows:
        raise RuntimeError(
            f"Vector index '{VECTOR_INDEX_NAME}' does not exist. "
            "Run `python run.py ingest` (it applies the schema first)."
        )
    config = (rows[0].get("options") or {}).get("indexConfig", {})
    actual = config.get("vector.dimensions")
    if actual is not None and int(actual) != int(dimensions):
        raise RuntimeError(
            f"Vector index '{VECTOR_INDEX_NAME}' was built for {actual} "
            f"dimensions but configs/base.yaml asks for {dimensions}.\n"
            "Neo4j will NOT error on a mismatched query - it returns nothing.\n"
            "Fix: `python run.py ingest --reset` to rebuild the index and "
            "re-embed the corpus."
        )
