# Cypher cookbook

**From "I have never seen a graph database" to the multi-hop query this project is built around.**

Everything here runs against the graph you build with `python run.py ingest`. Paste queries into the Neo4j Browser at <http://localhost:7474> (user `neo4j`, password `graphrag123`), or run the built-in ones with `python run.py cypher`.

---

## Part 0: the mental model, in 90 seconds

A graph database stores two things.

**Nodes** are things. A supplier, a component, a city.
**Relationships** are the connections between them, and they have a direction and a type.

Both can carry **properties** (key/value pairs), and nodes carry **labels** (their type).

```
(:Supplier {name: "Meridian Circuits"})  -[:SUPPLIES {sole_source: true}]->  (:Component {name: "PCB-A7"})
 ^label      ^property                     ^relationship type ^property         ^another node
```

Cypher is ASCII art of that picture:

- `()` is a node. `(s:Supplier)` is a node with the label `Supplier`, bound to the variable `s`.
- `-[]->` is a directed relationship. `-[:SUPPLIES]->` is one of type `SUPPLIES`.
- `-[]-` without an arrow is *undirected*: match it either way round.

If you know SQL, the useful comparison is:

| SQL | Cypher |
|---|---|
| `SELECT ... FROM` | `MATCH` |
| `WHERE` | `WHERE` |
| `JOIN suppliers ON ...` | `-[:SUPPLIES]->` |
| A join of unknown depth | `-[:DEPENDS_ON*1..3]->` **(this is the one SQL cannot really do)** |

That last row is the whole reason this project uses a graph database. Hold onto it.

---

## Part 1: your first five queries

Run these in order. Each one builds on the last.

### 1. What is even in here?

```cypher
MATCH (n)
RETURN labels(n) AS labels, count(*) AS count
ORDER BY count DESC
```

Expected: `Chunk` around 151, `Entity` around 82, `Document` 33, then the entity types.

> **Why `labels(n)` and not `n.label`?** A node can have several labels. Every entity here has two, `:Entity` plus its specific type, for reasons explained in [architecture.md](architecture.md).

### 2. Show me one supplier

```cypher
MATCH (s:Supplier {name: "Meridian Circuits Sdn Bhd"})
RETURN s
```

Click the node in the browser to expand its neighbours. This is the fastest way to build intuition about what a graph *is*.

### 3. What does that supplier sell us?

```cypher
MATCH (s:Supplier {name: "Meridian Circuits Sdn Bhd"})-[r:SUPPLIES]->(c:Component)
RETURN c.name AS component, r.sole_source AS sole_source, r.provenance AS source
```

| component | sole_source | source |
|---|---|---|
| PCB-A7 Main Controller Board | true | erp |
| PCB-B2 Sensor Interface Board | true | erp |

> `provenance: 'erp'` means this fact came from a structured export, not from a language model. Every relationship in this graph records which.

### 4. Two hops: which products contain those components?

```cypher
MATCH (s:Supplier {name: "Meridian Circuits Sdn Bhd"})-[:SUPPLIES]->(c:Component)<-[:CONTAINS]-(p:Product)
RETURN DISTINCT p.name AS product, c.name AS component
ORDER BY product
```

Note `<-[:CONTAINS]-` points *backwards*: the product contains the component, so the arrow runs from product to component and we are walking against it. Direction matters, and getting it wrong returns zero rows rather than an error.

### 5. The one SQL cannot do: variable depth

```cypher
MATCH path = (dependent:Supplier)-[:DEPENDS_ON*1..3]->(origin:Supplier)
RETURN dependent.name AS buyer,
       origin.name AS seller,
       length(path) AS hops
ORDER BY hops DESC, buyer
```

`*1..3` means "follow between one and three `DEPENDS_ON` relationships". You will see `Meridian Circuits → Formosa Substrate Materials` at 1 hop, and `Meridian Circuits → Sarawak Copper Foil` at 2 hops (through Formosa).

**You did not tell it how many hops to take.** That is the thing a `JOIN` cannot express, because a SQL join has a fixed shape written at query time. This is the moment graph databases justify themselves.

---

## Part 2: the queries this project is built on

### Products exposed to a location (the flagship)

```cypher
MATCH (loc:Location {key: "location:kaohsiung"})
MATCH (origin:Supplier)
WHERE (origin)-[:OPERATES]->(:Site)-[:LOCATED_IN]->(loc)
   OR (origin)-[:LOCATED_IN]->(loc)
MATCH depends = (dependent:Supplier)-[:DEPENDS_ON*0..3]->(origin)
MATCH (dependent)-[s:SUPPLIES]->(comp:Component)<-[:CONTAINS]-(prod:Product)
RETURN DISTINCT prod.name AS product, comp.name AS component,
       origin.name AS exposed_supplier, length(depends) AS tier_depth,
       [n IN nodes(depends) | n.name] AS chain
ORDER BY tier_depth, product
```

Three things to notice:

1. **`*0..3` includes zero hops.** A supplier that is itself in Kaohsiung is returned alongside deep-tier cases. Dropping the `0` hides the most obvious exposure of all - a classic off-by-one.
2. **The `WHERE` accepts two shapes.** Tier-1 suppliers come from CSV and always have a `Site`; tier-2 suppliers come from prose, where the model may or may not mint a `Site` node. Insisting on one shape would silently return half the exposure.
3. **`[n IN nodes(depends) | n.name]`** is a list comprehension over the path. Cypher has real collection operations, which is what makes returning a whole chain easy.

### Is our dual sourcing real?

```cypher
MATCH (sup:Supplier)-[:SUPPLIES]->(comp:Component {key: "part:li-18650-battery-pack"})
MATCH (sup)-[:DEPENDS_ON*1..3]->(upstream:Supplier)
WITH upstream, collect(DISTINCT sup.name) AS via
WHERE size(via) > 1
RETURN upstream.name AS shared_upstream, via AS reached_through
```

Result:

```
shared_upstream        reached_through
Baltic Lithium Salts   ["Nordcell Energi AB", "Volta Cell Systems"]
```

Two suppliers on two continents, one upstream source. **This is a set intersection.** No document in the corpus performs it, which is why no amount of searching finds it.

### Sole-sourced parts with an open finding

```cypher
MATCH (sup:Supplier)-[s:SUPPLIES]->(comp:Component)<-[:CONTAINS]-(prod:Product)
WHERE s.sole_source = true
OPTIONAL MATCH (f:Finding)-[:RAISED_AGAINST]->(sup)
WHERE f.status = 'open'
WITH sup, comp, collect(DISTINCT f.name) AS findings, collect(DISTINCT prod.name) AS products
WHERE size(findings) > 0
RETURN sup.name AS supplier, comp.name AS component, products, findings
```

`s.sole_source` comes from the ERP. `f.status` was extracted from an audit PDF by a language model. **This query joins a governed fact to an inferred one**, which is the practical argument for putting both in one database.

> Remove `WHERE f.status = 'open'` and the result includes suppliers whose only finding was a minor observation closed during the audit. A risk report that flags everybody flags nobody.

### Who would hurt us most?

```cypher
MATCH (sup:Supplier)-[:DEPENDS_ON*0..3]->(root:Supplier)
MATCH (sup)-[:SUPPLIES]->(:Component)<-[:CONTAINS]-(prod:Product)
WITH root, collect(DISTINCT prod.name) AS products
RETURN root.name AS supplier, size(products) AS products_at_risk, products
ORDER BY products_at_risk DESC
LIMIT 10
```

The top of that list contains companies Northwind has never placed an order with.

---

## Part 3: the text side

### Vector search

```cypher
CALL db.index.vector.queryNodes('chunk_embedding_index', 5, $embedding)
YIELD node AS chunk, score
MATCH (chunk)-[:PART_OF]->(doc:Document)
RETURN doc.doc_id, score, left(chunk.text, 120) AS preview
ORDER BY score DESC
```

You need a 768-float `$embedding` parameter, so this one is easier from the app than the browser. The point is that **vector search is just another Cypher call** here: no second database, no join key.

### The bridge

```cypher
MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity)
WHERE e.name = "Formosa Substrate Materials"
MATCH (c)-[:PART_OF]->(d:Document)
RETURN d.doc_id, m.confidence, left(c.text, 140) AS preview
```

This edge is what turns a vector hit into a graph anchor, and a traversal result back into a quotable chunk. It is the hinge of the whole architecture.

### Where did this edge come from?

```cypher
MATCH (a:Supplier)-[r:DEPENDS_ON]->(b:Supplier)
RETURN a.name AS buyer, b.name AS seller, r.confidence AS confidence,
       r.source_doc AS document, r.evidence AS sentence
```

Every LLM-extracted relationship stores the sentence it was based on. **An edge you cannot trace back to a sentence is an edge you cannot defend.**

---

## Part 4: indexes, and why queries are fast

```cypher
SHOW INDEXES;
SHOW CONSTRAINTS;
```

| Object | Purpose |
|---|---|
| `entity_key` (constraint) | Uniqueness **and** the lookup path for every `MERGE` in ingestion. Without it, ingestion goes quadratic. |
| `chunk_embedding_index` (vector) | HNSW approximate nearest neighbour over 768-dim embeddings |
| `entity_name_index` (full-text) | Entity linking: question text to graph nodes |
| `chunk_text_index` (full-text) | BM25 keyword retrieval |

### See the planner's mind

```cypher
PROFILE
MATCH (s:Supplier)-[:SUPPLIES]->(c:Component)
WHERE s.name = "Meridian Circuits Sdn Bhd"
RETURN c.name
```

Look at the first row of the plan. `NodeIndexSeek` means it used an index. `NodeByLabelScan` followed by a filter means it read every supplier and threw most away. On 13 suppliers you will not notice; on 13 million you will.

Now try the anti-pattern:

```cypher
PROFILE
MATCH (n:Entity)
WHERE n.type = "Supplier"
RETURN count(n)
```

versus

```cypher
PROFILE
MATCH (n:Supplier)
RETURN count(n)
```

Compare `db hits`. **A label is an index. A property equal to a string is not.** That is why every entity in this graph carries two labels.

---

## Part 5: things that will bite you

| Trap | What happens | Fix |
|---|---|---|
| Wrong relationship direction | Zero rows, **no error** | Use `-[:REL]-` undirected while exploring, then tighten |
| Unbounded `*` | Query never returns, or the heap dies | Always bound it: `*1..3` |
| Vector dimension mismatch | Empty results, **no error** | `python run.py doctor` checks this |
| String-interpolating values | Injection, plus a plan-cache miss on every call | Always use `$parameters` |
| `MERGE` on a non-indexed property | Ingestion goes quadratic | Add a uniqueness constraint first |
| `MERGE (a)-[:R]->(b)` with unbound `a`/`b` | Creates duplicate nodes | `MATCH` both first, then `MERGE` the relationship |

### The single most useful debugging habit

When a traversal returns nothing, **walk it backwards one hop at a time**:

```cypher
MATCH (l:Location {key:"location:kaohsiung"}) RETURN l                      // does the node exist?
MATCH (l:Location {key:"location:kaohsiung"})<-[:LOCATED_IN]-(x) RETURN x   // anything pointing at it?
MATCH (l:Location {key:"location:kaohsiung"})<-[:LOCATED_IN]-()<-[:OPERATES]-(s) RETURN s
```

The hop where the rows disappear is the hop where your data or your direction is wrong. This finds the problem faster than staring at the full query, every time.

---

## Where next

- [architecture.md](architecture.md) - why the schema is shaped this way
- [`src/graph/queries.py`](../src/graph/queries.py) - every query the app runs, with its reasoning
- `python run.py cypher` - the cookbook executed against your live graph
