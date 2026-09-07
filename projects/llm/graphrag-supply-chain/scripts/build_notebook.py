#!/usr/bin/env python
"""Build notebooks/GraphRAG_Supply_Chain.ipynb.

    python scripts/build_notebook.py

The notebook is generated rather than hand-edited so its code stays in sync
with the modules it imports, and so a JSON escaping mistake cannot ship a file
Jupyter refuses to open. Editing the notebook by hand is fine; regenerating
overwrites it.

The notebook does NOT duplicate the application. It imports the same modules
the app uses, so there is one implementation and the notebook is a lens onto
it. Cells are ordered so the ones needing no API key and no database come
first.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "GraphRAG_Supply_Chain.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md("""
# GraphRAG Supply Chain: concepts and experiments

This notebook is the **lens**, not the implementation. Every cell imports the
same modules the application uses, so there is exactly one version of the logic
and nothing here can drift from what ships.

**Order matters.** Parts 1 and 2 need no API key and no database. Part 3 onward
needs `python run.py ingest` to have been run.

| Part | Needs |
|---|---|
| 1. Chunking | nothing |
| 2. Entity resolution | nothing |
| 3. The graph | Neo4j + an ingested graph |
| 4. Retrieval compared | Neo4j + API key |
| 5. Where GraphRAG loses | Neo4j + API key |
| 6. Evaluation | Neo4j + API key |
""")

code("""
import sys, os
from pathlib import Path

# The notebook lives in notebooks/, the code in ../src
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.config import get_config
config = get_config()
print("project root:", ROOT)
print("model:       ", config.llm["model"])
print("embedding:   ", config.embedding["model"], config.embedding["dimensions"], "dims")
""")

# --- Part 1 ----------------------------------------------------------------
md("""
---
## Part 1: chunking, and the bug that hides three layers away

Chunking looks like boilerplate. It is the step most likely to silently destroy
your graph.

If a chunk boundary falls in the middle of *"Meridian purchases the copper-clad
laminate from Formosa Substrate Materials"*, then no chunk states that
relationship. The extractor never sees it, **no edge is created**, and the
multi-hop query returns nothing. The symptom appears in retrieval results,
three layers away from the cause.
""")

code("""
from src.ingest.chunker import load_documents, chunk_documents

documents = load_documents(config.documents_dir)
chunks = chunk_documents(documents, **{k: config.chunking[k] for k in
                                       ("chunk_size", "chunk_overlap", "min_chunk_chars")})

print(f"{len(documents)} documents -> {len(chunks)} chunks")
print(f"average {sum(len(c.text) for c in chunks) // len(chunks)} chars per chunk")
print()
print("Why the corpus size matters: with top_k =", config.retrieval["vector_top_k"],
      f"retrieval discards {100 * (1 - config.retrieval['vector_top_k'] / len(chunks)):.0f}% of what it could return.")
print("On a 4-document corpus, top-5 returns almost everything and every strategy")
print("scores identically - which is how a lot of published RAG comparisons")
print("end up measuring noise.")
""")

code("""
# Every chunk carries its document title and heading. That prefix costs ~15
# tokens and makes both the embedder and the extractor aware of what they are
# reading. A chunk starting "Corrective action requested:" is unattributable
# on its own.
print(chunks[40].text[:400])
""")

code("""
# The load-bearing check: do the six sentences that state a sub-tier dependency
# survive chunking intact? If any is split, that edge is silently lost.
required = [
    ("SUP-PROFILE-MERIDIAN", "purchased from Formosa Substrate Materials"),
    ("SUP-PROFILE-VOLTA",    "from Baltic Lithium Salts"),
    ("SUP-PROFILE-NORDCELL", "from Baltic Lithium Salts"),
    ("SUB-TIER-FORMOSA",     "Sarawak Copper Foil"),
    ("SUP-PROFILE-HELIOS",   "from Anhui Rare Earth Refining"),
    ("SUP-PROFILE-KAIGAN",   "from Kaohsiung Precision Glass"),
]
by_doc = {}
for chunk in chunks:
    by_doc.setdefault(chunk.doc_id, []).append(" ".join(chunk.text.split()))

for doc_id, phrase in required:
    ok = any(phrase in text for text in by_doc[doc_id])
    print(f"  {'OK  ' if ok else 'LOST'} {doc_id:<26} {phrase}")
""")

# --- Part 2 ----------------------------------------------------------------
md("""
---
## Part 2: entity resolution

The hardest correctness problem in GraphRAG, and it fails **silently in both
directions**:

- Split one supplier into three nodes and every traversal returns a third of the truth.
- Merge a city into a company and the system reports that a glass processor was
  hit by a typhoon that hit a city - **citing real evidence** for a false identity.

Nothing in the resolver is a language model. It runs on every extracted mention,
so it must be fast, free, and above all **deterministic**: a resolver that
returns different answers on different runs makes the whole graph
irreproducible.
""")

code("""
from src.ingest.resolve import EntityResolver, make_key, normalise

# Stage 1: normalisation. Legal suffixes carry no identity.
for name in ["Helios Fluidics BV", "Helios Fluidics", "Meridian Circuits Sdn Bhd",
             "Skelleftea", "Skellefte\\u00e5"]:
    print(f"  {name:<30} -> {normalise(name)!r}")
""")

code("""
resolver = EntityResolver()
resolver.register("Supplier", "Meridian Circuits Sdn Bhd",
                  aliases=["Meridian Circuits", "Meridian"], authoritative=True)
resolver.register("Component", "DSP-3300 5.5in TFT Display Module",
                  aliases=["DSP-3300"], authoritative=True)
resolver.register("Location", "Kaohsiung", authoritative=True)

tests = [
    ("Supplier", "Meridian",                    "alias -> canonical"),
    ("Supplier", "Meridian Circuits",           "suffix variation"),
    ("Product",  "DSP-3300",                    "WRONG TYPE from the model; the ERP wins"),
    ("Supplier", "Kaohsiung Precision Glass",   "must NOT merge with the city"),
    ("Location", "Kaohsiung, Taiwan",           "city, country -> city"),
]
for etype, name, why in tests:
    entity = resolver.resolve(etype, name)
    print(f"  {name:<30} -> {entity.key:<34} ({entity.type})   {why}")

print()
print("resolver stats:", resolver.stats)
print()
print("'retyped' is the interesting one. Asked to label DSP-3300 in the sentence")
print("'the DSP-3300 display module used in the NW-500', Product is a reasonable")
print("reading. But the PLM says it is a Component, and the PLM is authoritative")
print("about its own parts. Without that rule the graph grows a phantom Product")
print("that no bill of materials contains and no traversal can reach.")
""")

code("""
# Determinism: two resolvers, same input order, must agree exactly.
def build():
    r = EntityResolver()
    r.register("Supplier", "Volta Cell Systems", aliases=["Volta"], authoritative=True)
    return [r.resolve("Supplier", n).key for n in
            ["Volta", "Volta Cell", "Baltic Lithium Salts", "volta cell systems"]]

print("run 1 == run 2:", build() == build())
""")

# --- Part 3 ----------------------------------------------------------------
md("""
---
## Part 3: the graph

**From here on you need Neo4j running and `python run.py ingest` completed.**

```bash
docker compose up -d
python run.py ingest
```
""")

code("""
from src.graph.client import GraphClient
from src.graph import queries

client = GraphClient(config)
client.verify()

counts = client.counts()
for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {label:<26} {n:>5}")
""")

code("""
# Every sub-tier dependency, with the sentence it was extracted from.
# This is the difference between a knowledge graph and a rumour graph.
rows = client.run(\"\"\"
MATCH (a:Supplier)-[r:DEPENDS_ON]->(b:Supplier)
RETURN a.name AS buyer, b.name AS seller, r.confidence AS confidence,
       r.source_doc AS document, r.evidence AS sentence
ORDER BY buyer
\"\"\")
for row in rows:
    print(f"{row['buyer']}  ->  {row['seller']}   (confidence {row['confidence']})")
    print(f"    [{row['document']}] \\"{(row['sentence'] or '')[:110]}...\\"")
    print()
""")

md("""
### The flagship traversal

No document in the corpus contains this answer. The incident bulletin explicitly
says so. It exists only as a join across five relationship types.
""")

code("""
rows = client.run(queries.PRODUCTS_EXPOSED_TO_LOCATION,
                  location_key="location:kaohsiung", limit=25)
print(f"{len(rows)} exposure paths through Kaohsiung\\n")
for row in rows:
    chain = " -> ".join(reversed(row["dependency_chain"]))
    print(f"  tier {row['tier_depth']}  {row['product']:<26} via {row['component']:<34} [{chain}]")

print()
print("Four products. Two of them exposed twice, through independent paths.")
""")

code("""
# The dual-sourcing illusion: a set intersection no document performs.
rows = client.run(queries.SHARED_UPSTREAM_FOR_COMPONENT,
                  key="part:li-18650-battery-pack")
for row in rows:
    print(f"  {row['component']}")
    print(f"    shared upstream : {row['shared_upstream']}")
    print(f"    reached through : {', '.join(row['reached_through'])}")
print()
print("Two suppliers, two continents, one upstream source. Which suppliers")
print("supply the part is an ERP fact; what each buys upstream is in two")
print("separate documents. Nothing in the corpus joins them.")
""")

# --- Part 4 ----------------------------------------------------------------
md("""
---
## Part 4: retrieval strategies, compared

**Needs an API key from here on.**

The same question through every strategy. Watch the *evidence*, not just the
answer: which documents each strategy found is usually the clearest view of why
one wins.
""")

code("""
from src.llm import LLMClient
from src.retrieval.strategies import Retriever, STRATEGIES, STRATEGY_LABELS

llm = LLMClient(config)
retriever = Retriever(client, llm, config)

QUESTION = ("Typhoon Meilin has shut down Kaohsiung for about three weeks. "
            "Which of our finished products are exposed, and through which suppliers?")

# Warm the embedding cache BEFORE timing anything. Without this the first
# strategy pays the API round trip and every later one hits the disk cache,
# which made dense retrieval look 7x slower than a strategy doing strictly
# more work. That is a reversal, and it would be invisible in the output.
llm.embed_query(QUESTION)

results = {s: retriever.retrieve(QUESTION, s) for s in STRATEGIES}

print(f"{'strategy':<10} {'chunks':>7} {'graph':>7} {'entities':>9} {'hops':>5} {'ms':>7}")
print("-" * 50)
for name, result in results.items():
    print(f"{name:<10} {len(result.text_evidence):>7} {len(result.graph_evidence):>7} "
          f"{len(result.entities):>9} "
          f"{max([e.hops for e in result.entities], default=0):>5} "
          f"{result.latency_ms:>7.1f}")
""")

code("""
# Which documents did each strategy find that the others did not?
doc_sets = {name: {e.doc_id for e in r.text_evidence if e.doc_id}
            for name, r in results.items()}
every = sorted(set().union(*doc_sets.values()))

print(f"{'document':<32} " + " ".join(f"{s[:6]:>7}" for s in STRATEGIES))
print("-" * 78)
for doc in every:
    marks = " ".join(f"{'yes' if doc in doc_sets[s] else '.':>7}" for s in STRATEGIES)
    print(f"{doc:<32} {marks}")
""")

code("""
# The derived facts: computed by traversal, quoted from no document.
for item in results["hybrid"].graph_evidence:
    print(item.text[:900])
    print()
""")

code("""
# How hybrid got there, in its own words.
for i, line in enumerate(results["hybrid"].trace, 1):
    print(f"{i}. {line}")
""")

# --- Part 5 ----------------------------------------------------------------
md("""
---
## Part 5: where GraphRAG loses

This is the part most GraphRAG material skips, and it is the part that turns a
demo into engineering judgement.

Some questions are answered inside one document. For those, traversal is pure
cost: more latency, more context, same answer. And **graph-only retrieval fails
outright** on a question that names no entity, because it has nowhere to start.
""")

code("""
from src.answer import AnswerEngine
engine = AnswerEngine(llm, config)

LOSING = "Under our dual sourcing policy, what makes a component require a second source?"
llm.embed_query(LOSING)

for name in ["vector", "graph", "hybrid"]:
    result = retriever.retrieve(LOSING, name)
    answer = engine.answer(LOSING, result)
    docs = sorted({e.doc_id for e in result.text_evidence if e.doc_id})
    print(f"--- {STRATEGY_LABELS[name]} ---")
    print(f"    documents: {docs or 'NONE'}")
    print(f"    latency:   {result.latency_ms:.0f} ms retrieval")
    print(f"    answer:    {answer.text[:180].strip()}...")
    print()

print("graph-only retrieves nothing: it links no entity, so it has nowhere to")
print("start traversing. That is the honest cost of throwing away the vector arm,")
print("and it is why the shipped architecture is hybrid rather than pure graph.")
""")

# --- Part 6 ----------------------------------------------------------------
md("""
---
## Part 6: evaluation

A benchmark containing only questions your system wins is marketing. Five of the
twelve golden questions are ones GraphRAG should **not** win.

The full run is 12 questions x 5 strategies and takes several minutes. This cell
runs a subset with the LLM judge off, so the metrics are fully deterministic.
""")

code("""
from src.evaluate import evaluate, format_report

report = evaluate(
    strategies=["vector", "classic", "graph", "hybrid"],
    question_ids=["GQ-01", "GQ-03", "GQ-04"],   # multi-hop, single-doc, definitional
    judge=False,
    config=config,
)
print(format_report(report))
""")

md("""
Read the **by category** table, not the overall one.

- On `multi_hop`, hybrid should lead clearly. That is the claim.
- On `single_document` and `definitional`, vector should **match** hybrid. A
  GraphRAG system that broke plain lookups to win multi-hop would be a bad trade.
- `graph` alone should score zero on `definitional`. It has nowhere to start.

If the graph strategies win every category, the baseline is broken and every
other number is worthless.
""")

# --- Part 7 ----------------------------------------------------------------
md("""
---
## Part 7: your turn

Things worth trying:

1. **Add a document.** Drop a `.md` file into `data/documents/` with front
   matter, re-run `python run.py ingest`, and watch the resolver stats. Does it
   merge onto existing entities or create new ones?
2. **Break the dimensions.** Set `embedding.dimensions: 1536` in
   `configs/base.yaml` *without* re-ingesting. `run.py doctor` catches what
   would otherwise be a silent empty-result bug.
3. **Poison the graph.** Read `data/adversarial/POISONED-SUPPLIER-RESPONSE.md`,
   then run the cell below to watch the guardrail block it. Try rewording the
   payload until it gets through - that is the most instructive exercise here,
   and it is why the README says detection is best-effort and traceability is
   the real control.
4. **Write a Cypher query** the templates do not cover, using
   `docs/cypher-cookbook.md`.
""")

code("""
from src.guardrails.injection import scan_document

payload = (ROOT / "data" / "adversarial" / "POISONED-SUPPLIER-RESPONSE.md").read_text(encoding="utf-8")
result = scan_document(payload, "POISONED")

print("blocked:", result.blocked)
print("summary:", result.summary())
print()
for d in result.detections:
    print(f"  [{d.severity}] {d.group}")
    print(f"      \\"{d.excerpt[:110]}...\\"")

print()
print("In ordinary RAG this payload corrupts one answer.")
print("In GraphRAG it would write three fabricated DEPENDS_ON edges into shared,")
print("persistent state, delete a real one, and every future traversal would")
print("reach them - carrying a real citation, because the sentence really is in")
print("the document.")
""")

code("""
client.close()
print("done")
""")


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
