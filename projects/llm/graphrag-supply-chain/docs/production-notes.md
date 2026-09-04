# Production notes

**What is deliberately simple here, what would change, and why.**

The runnable project is not under-engineered by accident. Every simplification below was a choice, and each one is listed with the condition that would force the change. A learning project that quietly ships production complexity teaches the complexity instead of the idea; one that pretends the simplifications are sufficient teaches something worse.

---

## The general rule

> Build the simple thing until you can state the specific condition that breaks it. Then build the next thing, for that condition.

Most of the items below are not "we cut corners". They are "this is correct at 33 documents and one process, and here is exactly when it stops being correct".

---

## 1. Ingestion

| | Here | Production | Trigger to change |
|---|---|---|---|
| Trigger | On demand, full rebuild | CDC from ERP + a document queue | More than a few hundred documents, or any SLA on freshness |
| Granularity | Everything, every time | Incremental by content hash | Re-ingest cost becomes noticeable |
| Extraction unit | Whole document | Windowed, with a carried-forward entity summary | Documents exceed the context window |
| Concurrency | One job, in-process lock | Queue with a single consumer per graph | More than one API replica |
| Failure | Skip the document, record it | Dead-letter queue + human review | Any volume at all |

### Whole-document extraction has a ceiling

The choice to extract per document rather than per chunk is correct here and will not scale. A 400-page contract does not fit, and the fix is not simply chunking it: relationships span sections, which is the whole reason document-level extraction was chosen.

The production pattern is a sliding window that carries forward a **summary of entities found so far**, so a window discussing "its only qualified source" still knows whose source it is. That summary is itself model output and can drift, which is why the windows should overlap and the results should be de-duplicated by the same resolver used everywhere else.

### Re-embedding is a migration, not a config change

Changing `embedding.model` or `embedding.dimensions` invalidates every vector in the database. Doing it in place gives you a window where some chunks are in the old space and some in the new, and **similarity between them is meaningless without erroring**. Retrieval quality degrades and nothing alerts.

The safe pattern: build a second vector index alongside the first, backfill it, verify, swap the index name, then drop the old one. The embedding cache key already includes model and dimensions, so a re-embed never silently reuses stale vectors.

---

## 2. Entity resolution

The three-stage deterministic ladder is the right hot path and should stay. What production adds is **stage 4: adjudication**.

`rejected_fuzzy` in the ingest report counts pairs that scored above 0.75 and below the 0.90 threshold. Those are the genuinely ambiguous ones - the pairs where a human, or an LLM, would actually add value:

```
"Formosa Substrate Materials"  vs  "Formosa Substrate Materials Co., Ltd."   -> same
"Formosa Substrate Materials"  vs  "Formosa Chemical Industries"             -> different
```

The production design is a **review queue**, not an inline model call:

1. Deterministic ladder runs inline, as now.
2. Rejected-fuzzy pairs go to a queue with both surface forms, their source documents and their neighbourhoods.
3. An LLM proposes a verdict with a rationale.
4. A human confirms or overrides. The decision is written to an **alias table** that becomes authoritative.
5. Stage 2 consults that table on every subsequent run.

The critical property is that resolution stays deterministic at ingest time. An LLM in the inline path makes the graph irreproducible: two runs over identical input produce different node identities, and nothing downstream can be compared across runs.

### The failure mode nobody plans for

Entity resolution quality **degrades as the corpus grows**, because the chance of two genuinely different entities having similar names rises with the number of entities. A threshold tuned on 80 entities will over-merge at 80,000. Monitor `rejected_fuzzy` and merge rates over time; a step change means the threshold needs revisiting, not that the corpus got messier.

---

## 3. Retrieval

### Text-to-Cypher, and when it earns its place

Template routing is used here because it is stable, bounded, and cannot surprise you. It has a real cost: a question the templates do not cover gets generic neighbourhood expansion instead of a precise query.

Text-to-Cypher becomes worth it when:

- the query space is genuinely open-ended (analysts asking novel questions), **and**
- there is a human reading the result who can spot a wrong query, **and**
- you can bound the damage.

Bounding it is the part usually skipped:

- run generated queries as a **read-only Neo4j user**,
- enforce a server-side query timeout,
- reject any query without a `LIMIT`, or inject one,
- reject unbounded variable-length patterns (`*` with no upper bound),
- show the generated Cypher to the user before or alongside the answer.

The failure mode that makes this dangerous is not a query that errors. It is a query that **returns rows that are subtly wrong**, which reads as a confident answer.

### Reranking

Not included, deliberately. A cross-encoder reranker over the top 20 candidates is a standard and usually worthwhile addition - but on a 151-chunk corpus, a top-20 fetch returns 13% of everything, so the reranker would have almost nothing to discriminate. Adding it here would produce a component that looks impressive and cannot be shown to help, which is worse than not having it.

Add it when the corpus is large enough that top-`k` is a genuine filter and dense retrieval is returning near-misses at ranks 6 to 20.

### Caching

Answer caching is tempting and dangerous in this system. A derived graph fact computed for one caller **must not** be served to another with different entitlements. Cache keys must include the caller's entitlement set, or the cache must hold only the pre-authorisation retrieval and recompute the filtered view per request.

---

## 4. Serving and scale

| Component | Here | At scale |
|---|---|---|
| API | One uvicorn process | Multiple replicas behind a load balancer |
| Rate limiting | In-process sliding window | Redis, or the ingress |
| Job state | In-process objects | Redis or a database, so status survives a restart |
| Neo4j | Single container, 1 GB heap | Causal cluster; read replicas for query traffic |
| Embedding cache | Local pickle files | Shared cache keyed by content hash |

**The in-process rate limiter is the item most likely to be forgotten.** Two replicas means two independent windows and exactly twice the intended allowance, while the config, the health check and the code all still say the limiter is enabled. Lifetime is part of a control's correctness.

### Neo4j sizing, briefly

The page cache should hold the working set. For a graph like this one, that is the node and relationship stores plus the indexes; the vector index is the memory-hungry part, at roughly `dimensions × 4 bytes × chunks` plus HNSW overhead. At 768 dimensions and a million chunks that is around 3 GB before overhead, which is the number that decides your instance size.

---

## 5. Observability

The audit log covers *what happened*. Production wants *how it is behaving*:

| Signal | Why it matters |
|---|---|
| Latency **per stage** (embed / vector / traverse / generate) | A slow answer is usually one stage, and without the split you will optimise the wrong one |
| Cost per request, per user, per day | The first symptom of an unbounded loop is the invoice |
| **Retrieval quality over time** | Run the golden set on a schedule against production data. Retrieval degrades silently as a corpus grows. |
| Guardrail firing rates | A rate that drops to zero after a deploy means a broken control, not a safer world |
| `rejected_fuzzy` trend | Early warning on entity resolution drift |
| Empty-retrieval rate | The clearest single indicator that something upstream broke |

The last one deserves emphasis. **Unwarranted refusal is a retrieval failure wearing a polite hat**, and it is invisible in user-satisfaction metrics because the answer sounds responsible. It is measured explicitly in the evaluation for that reason and should be a production alert.

---

## 6. Cost control

Measured on this corpus: full ingestion is ~$0.06, a hybrid question ~$0.0006, the full evaluation ~$0.08.

What keeps it there:

| Lever | Effect |
|---|---|
| Thinking **off** for extraction | Extraction is transcription against a closed vocabulary. Thinking multiplied the token bill and broke it entirely at 2048/8192 budgets. |
| Embedding cache keyed by content | Re-ingestion after a code change costs nothing for embeddings |
| Mentions linked by string matching | 151 chunks × an LLM call each, avoided entirely |
| Document-level extraction | 33 calls instead of ~200 |
| Per-request budget cap | Bounds the blast radius of a loop or a malformed input |

At production scale add: extraction only for changed documents (by content hash), a smaller model for extraction than for answering if quality holds, and batching where the provider supports it.

---

## 7. Data quality, which is the real long-term risk

The failure that will actually hurt a deployed version of this system is not a crash. It is **quiet graph decay**:

- a supplier renamed in the ERP becomes a second node,
- an extraction regression drops a relationship type and nobody notices for a month,
- a document is re-ingested with different content under the same id,
- a threshold tuned on a small corpus starts over-merging on a large one.

Every one of these returns *plausible* answers. Defences:

1. **Run the golden set on a schedule**, not only in CI. A drop in evidence recall is the earliest detectable signal.
2. **Assert invariants after every ingestion.** The project already does this - no chunk without an embedding, the vector index returns for a vector from itself, `MENTIONS` exist, `DEPENDS_ON` exist. Add domain ones: every component has a supplier, sole-source flags are consistent, share percentages sum to 100. Several of those are already unit tests over the CSVs.
3. **Version the graph.** Tag nodes with the ingestion run that created them, so a bad run can be identified and rolled back rather than reasoned about.
4. **Alert on structural deltas.** A rebuild that produces 20% fewer relationships than the last one is an incident, even if nothing errored.

---

## 8. What I would build next, in order

1. **Incremental ingestion by content hash.** Biggest cost and latency win, low risk.
2. **The entity-resolution review queue.** Highest quality-per-effort, and it compounds.
3. **Authorisation pushed into the Cypher templates.** The blocker for any real deployment, and the hardest item here.
4. **Scheduled golden-set runs with alerting.** Turns evaluation from a one-off claim into a monitor.
5. **Windowed extraction.** Needed the first time someone points this at real PDFs.
6. **Reranking.** Only once the corpus is large enough to measure whether it helps.

Note what is *not* on that list: more retrieval strategies, a bigger model, an agent loop. The evaluation says the retrieval architecture is not the bottleneck. Data quality and authorisation are.
