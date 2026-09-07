# Architecture

The technical reference. The README explains *why* the system exists; this explains *how it is put together* and where each decision lives in the code.

---

## 1. Layers

```
app/            UI            renders, decides nothing, talks HTTP only
api/routes_*    ROUTING       validates the request, calls ONE service, maps the result
src/services/   ORCHESTRATION what happens in what order, which guardrail runs when
src/            CAPABILITIES  retrieval, graph, llm, guardrails, ingest
```

**Enforced, not described.** [`tests/test_layering.py`](../tests/test_layering.py) fails the build if:

- anything under `src/` imports `fastapi`, `starlette` or `streamlit`,
- anything under `app/` imports `src` or `api`, or a database/model driver,
- a route module imports a capability directly instead of going through a service,
- a route handler exceeds 25 lines (a proxy for orchestration leaking downward),
- `run.py` stops calling `QAService.ask`.

### Why each boundary

| Boundary | If you cross it |
|---|---|
| UI must not hold logic | A guardrail enforced in a frontend is one anyone can skip with `curl` |
| Services must not import FastAPI | The pipeline becomes callable only from a web request, so the CLI, notebook and tests each grow their own copy, and the copies drift |
| Routes must not orchestrate | The same sequence exists twice, and a fix in one survives in the other |

The concrete payoff: `python run.py ask` and `POST /api/ask` are the same code path. The CLI gets rate limiting, budget caps, injection scanning and the audit log for free, and cannot fall behind the API.

### Composition root

[`api/deps.py`](../api/deps.py) is the only place the object graph is wired. Everything is built once at startup, because lifetime is a correctness property here:

| Object | Per-request construction would |
|---|---|
| `GraphClient` | Open a new Neo4j connection pool per request; exhausts the server in about a minute |
| `LLMClient` | Discard the embedding cache, so it never warms |
| `GuardrailEngine` | **Reset the rate limiter's sliding window**, so it permits unlimited requests while reporting itself enabled |

---

## 2. The data model

Two subgraphs, one bridge. Defined and commented in [`src/graph/schema.py`](../src/graph/schema.py).

### Text subgraph

```
(:Document {doc_id, title, doc_type, published})
   ^
   |  :PART_OF
(:Chunk {chunk_id, text, embedding, ord})
```

### Knowledge subgraph

```
(:Product)  -[:CONTAINS {quantity}]->  (:Component)
(:Supplier) -[:SUPPLIES {sole_source, share_pct}]-> (:Component | :Material)
(:Supplier) -[:DEPENDS_ON]->  (:Supplier)          <- the tier-2/3 gold
(:Supplier) -[:OPERATES]->    (:Site) -[:LOCATED_IN]-> (:Location)
(:Incident) -[:AFFECTS]->     (:Site | :Supplier | :Location)
(:Finding {status}) -[:RAISED_AGAINST]-> (:Supplier)
(:Regulation) -[:APPLIES_TO]-> (:Component | :Material)
(:Supplier) -[:HOLDS]->       (:Certification)
```

### The bridge

```
(:Chunk)-[:MENTIONS {confidence}]->(:Entity)
```

Run forwards it turns a vector hit into graph anchors. Run backwards it turns traversal results into quotable chunks. **Text → structure → text.** That round trip is the mechanism by which the system retrieves evidence that is not similar to the question.

### Double labelling

Every entity is created as `(:Entity:Supplier)`, `(:Entity:Component)` and so on.

| Label | Buys |
|---|---|
| `:Entity` | One uniqueness constraint, one full-text index, one generic traversal for all ten types |
| `:Supplier` | Label-scoped scans. **A label is an index; `WHERE n.type = 'X'` is not.** |

Cost: one extra label per node and the discipline of writing both at creation. `upsert_entities()` is the single place that happens.

### Node keys

```
supplier:meridian-circuits
part:li-18650-battery-pack
location:kaohsiung
```

The key is `{namespace}:{normalised-name}`. Two design points:

1. **Type is in the key**, so a `Location` "Meridian" and a `Supplier` "Meridian" cannot collide under the uniqueness constraint and become indistinguishable forever.
2. **`Component` and `Material` share the `part:` namespace.** A copper-clad laminate is legitimately both, and the distinction is a modelling convention rather than a fact about the world. Left separate, the corpus produces "chemically strengthened cover glass" twice, once under each label, and every traversal through it returns half the truth.

### Provenance on every relationship

| Property | Meaning |
|---|---|
| `provenance` | `erp` (structured export) or `llm` (extracted from prose) |
| `confidence` | 1.0 for ERP; the extractor's score for LLM |
| `evidence` | The **verbatim sentence** the edge was based on (LLM only) |
| `source_doc` | Which document |

This is what lets a user distinguish a governed fact from a model's reading, and what makes a bad edge findable rather than arguable. The UI draws ERP edges solid and extracted edges dashed.

---

## 3. Indexes

| Index | Type | Purpose |
|---|---|---|
| `entity_key` | Uniqueness constraint | Identity **and** the `MERGE` lookup path. Without it ingestion goes quadratic. |
| `document_id`, `chunk_id` | Uniqueness constraints | Idempotent re-ingestion |
| `entity_type`, `chunk_doc` | Range | Filter pushdown for UI panels |
| `chunk_embedding_index` | **Vector (HNSW)** | Dense retrieval, 768 dims, cosine |
| `entity_name_index` | **Full-text** | Entity linking over `name` + `aliases` |
| `chunk_text_index` | **Full-text** | BM25 keyword retrieval |

### Two full-text indexes, two different jobs

`entity_name_index` handles **entity linking**, which is a *lexical* problem: the user typed the name. Lucene does it better and roughly 200x cheaper than an embedding round trip. Using vector similarity for name lookup is a real and common mistake - it cheerfully returns "Kaohsiung Precision Glass" when you asked about the city, because those strings are semantically close.

`chunk_text_index` gives the honest baseline its keyword arm. Part numbers like `PCB-A7` carry almost no distributional meaning, so embeddings are bad at them.

### `apply_schema` waits for indexes

Index creation in Neo4j is asynchronous. Querying a vector index that is still `POPULATING` returns **zero rows and no error**, which during ingestion looks exactly like "the embeddings did not save". `apply_schema()` calls `db.awaitIndexes()`, removing a whole class of phantom bug.

---

## 4. Ingestion pipeline

[`src/ingest/pipeline.py`](../src/ingest/pipeline.py). Order is load-bearing.

| # | Step | Why here |
|---|---|---|
| 1 | Schema | Every `MERGE` below needs an index to seek on |
| 2 | Backbone (CSV) | **Registers ERP names first**, so they win every identity contest |
| 3 | Chunk | Structure-aware, heading boundaries first |
| 4 | **Guardrail** | Before the extractor, because extractor output becomes persistent shared state |
| 5 | Extract | One LLM call per document |
| 6 | Embed | Cached by `(text, task, model, dims)` |
| 7 | Write | `UNWIND`-batched |
| 8 | **Verify** | Assert the graph is queryable before declaring success |

Step 2's position is subtle and important. Because the ERP names are registered before any document is read, a later mention of "Meridian" in prose resolves *onto* the canonical node instead of founding a rival one that happens to be first.

Step 8 is the one usually omitted. It caught the zero-extraction bug during development, surfacing it as `no DEPENDS_ON relationships were extracted` rather than as a plausible but empty graph.

---

## 5. Retrieval

[`src/retrieval/strategies.py`](../src/retrieval/strategies.py). All five strategies return the same `RetrievalResult`, which is what makes the comparison honest: same prompt, same citation format, same measurement, one variable.

### Hybrid, in order

1. Vector search, `2k` candidates.
2. BM25, `2k` candidates.
3. **Entity linking** from the question text (full-text over names and aliases).
4. **Chunk-mention anchoring** from the top vector hits via `MENTIONS`.
5. Traversal from the union of 3 and 4, up to `max_hops`.
6. Type-driven templates fire on the linked entity types.
7. Derived facts rendered as labelled blocks.
8. Text lists fused with RRF to `k`; derived facts prepended, **not** fused.

Steps 3 and 4 exist together because they fail differently. Name matching fails when the question names nothing ("which products are exposed to the typhoon"); chunk anchoring fails when vector search lands on the wrong document. Together they are considerably more robust.

Step 8's asymmetry is deliberate: a derived fact is not competing with text for a slot. It is a different *kind* of evidence, and ranking a computed join against a cosine score would be comparing incommensurable things.

### Traversal safety

- Restricted to `KNOWLEDGE_RELS`, so an expansion can never wander through `MENTIONS` and conclude that two suppliers are related because one PDF named both. That is a coincidence, not a relationship, and letting it into a path is how a GraphRAG system starts producing confident nonsense.
- Depth validated by `_depth()` to `1..5` before being formatted into a query string. It is the only value that cannot be a parameter, so it is the only one that must be proven safe.
- Every traversal carries a `LIMIT`. Undirected expansion at depth 3 through a hub node is combinatorial.

---

## 6. Answer generation

[`src/answer.py`](../src/answer.py) enforces three properties: **grounding** (context only), **attribution** (a citation per claim, derived facts cited as derived), and **refusal** (saying "I don't know" is a correct output).

Context assembly puts derived facts first. Models attend more reliably to the start of a long context, and burying a three-line exposure path under 8,000 characters of prose is a measurable way to lose the answer the traversal was run to get.

If retrieval returns nothing, the engine refuses **without a model call**. That keeps the refusal wording deterministic, which the evaluation depends on, and avoids paying to be told there is nothing to say.

---

## 7. Where each concept lives

| Concept | File |
|---|---|
| Graph schema, constraints, indexes | `src/graph/schema.py` |
| Every Cypher query, with reasoning | `src/graph/queries.py` |
| Batched writes, connection handling | `src/graph/client.py` |
| Structure-aware chunking | `src/ingest/chunker.py` |
| Structured extraction, closed vocabulary | `src/ingest/extract.py` |
| Entity resolution ladder | `src/ingest/resolve.py` |
| Structured/unstructured split | `src/ingest/loader.py` |
| RRF fusion, evidence types | `src/retrieval/base.py` |
| The five strategies, query planning | `src/retrieval/strategies.py` |
| Injection defence | `src/guardrails/injection.py` |
| Output validation | `src/guardrails/validate.py` |
| Budgets and rate limits | `src/guardrails/limits.py` |
| Orchestration | `src/services/` |
| Benchmark harness | `src/evaluate.py` |
| Provider boundary | `src/llm.py` |
