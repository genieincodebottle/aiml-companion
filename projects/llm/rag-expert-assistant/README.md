# RAG Expert Assistant

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Production RAG system with chunking, reranking, security, and evaluation

## Problem Statement

Naive LLM applications hallucinate, ignore context, and leak PII. This project builds a **production-grade RAG pipeline** that grounds answers in retrieved documents, validates retrieval quality with RAGAS metrics, defends against prompt injection, and provides an A/B framework for measuring optimization impact.

## Architecture

```
Documents (PDF/MD/TXT)
    |
    v
+----------------------------------------------+
|  Ingestion Pipeline                           |
|  Load -> Chunk (512 tokens, 50 overlap)       |
|  -> Embed (gemini-embedding-001) -> ChromaDB  |
+----------------------------------------------+
    |
    v
+--------------------------+  +---------------------+
|  Retrieval + Reranking    |  |  Security Layer      |
|  Top-20 candidates        |  |  PII Detection       |
|  -> FlashRank -> Top-5    |  |  Injection Defense   |
+--------------------------+  |  Output Filtering    |
    |                          +---------------------+
    v                                   |
+--------------------------+            |
|  Generation (Gemini 3.5   |<----------+
|  Flash Lite)              |
|  Grounded prompt          |
|  + Citation extraction    |
+--------------------------+
    |
    v
+--------------------------+
|  Evaluation (RAGAS)       |
|  Faithfulness | Relevancy |
|  Precision | Recall       |
|  A/B: Naive vs Optimized |
+--------------------------+
```

## Results

### RAGAS Evaluation Scores

Measured over three runs of `python -m src.evaluate`, judged by
`gemini-3.5-flash-lite`. RAGAS is an LLM-judged metric, so the spread across
runs is part of the result and is reported here rather than hidden behind a
single decimal.

| Metric | Run 1 | Run 2 | Run 3 | Status |
|--------|-------|-------|-------|--------|
| Faithfulness | 0.938 | 0.938 | 0.938 | PASS |
| Answer Relevancy | 0.827 | 0.825 | 0.824 | NEEDS WORK (< 0.85) |
| Context Precision | 0.750 | 0.750 | 0.875 | UNSTABLE |
| Context Recall | 1.000 | 1.000 | 1.000 | PASS |

The previously published table (0.920 / 0.875 / 0.850 / 0.810) does not
reproduce. Context recall in particular was reported as the **weakest** metric
at 0.810; measured, it is the **strongest** at a perfect 1.000, and every piece
of "next steps" advice built on recall being the bottleneck was aimed at the
wrong metric.

**Read this table with its caveat.** `src/evaluate.py` scores a fixed dataset of
four hand-written question/answer/context examples. The contexts are typed into
the file, not retrieved by the pipeline. So `context_precision` here grades a
fixture, not the retriever, and this evaluation cannot tell you whether
chunking or reranking is working. For that, see the A/B below, which does run
the real pipeline.

### Naive vs Optimized RAG (A/B Comparison)

`src/ab_comparison.py` used to be a skeleton that invented its own
results: `evaluate_rag()` drew scores from `random.uniform` around a hardcoded
base of **0.65 for "Naive"** and **0.88 for "Optimized"**, so the optimized
config won by ~0.23 every run, formatted exactly like a real table. It now
builds both pipelines against the real corpus, runs the real questions, and
scores the real answers with RAGAS.

Measured over three runs, 10 questions per arm:

| Metric | Naive | Optimized | Delta | Runs agreeing |
|--------|-------|-----------|-------|---------------|
| Faithfulness | 0.983 / 1.000 / 1.000 | 0.913 / 0.980 / 0.942 | **-0.020 to -0.070** | WORSE in 3/3 |
| Answer Relevancy | 0.832 / 0.821 / 0.837 | 0.842 / 0.838 / 0.847 | +0.010 to +0.016 | BETTER in 3/3 |
| Context Precision | 0.950 / 0.950 / 0.950 | 0.950 / 0.950 / 0.950 | **0.000** | SAME in 3/3 |
| Context Recall | 0.967 / 0.967 / 0.967 | 1.000 / 1.000 / 1.000 | +0.033 | BETTER in 3/3 |

Optimized wins 2 of 4 metrics, loses faithfulness, and is roughly **4x slower**
(213 s vs 55 s per run). The old fabricated table claimed it won all four, with
"+24% faithfulness" and "+17% context precision".

**Why the optimizations do nothing here, and it is not because they are bad
ideas.** The demo corpus is 4 documents totalling **3,396 characters** -- about
one page of text. That splits into **5 chunks** under the naive config and
**9 chunks** under the optimized one. The optimized retriever asks for the top
20 candidates from a store containing 9, so stage one returns *the entire
corpus* and the reranker's job reduces to discarding 4 of 9 chunks. Two-stage
retrieval cannot improve precision when the first stage already returns
everything, which is exactly why context precision is identical to three
decimal places in all three runs: both arms score the same candidate pool.

Faithfulness drops because the optimized arm hands the model 5 chunks where the
naive arm hands it 3. On a corpus this small the extra chunks are not missing
evidence, they are distractors, and more text to blend means more opportunity
for an unsupported detail.

Reranking earns its latency at corpus scale, where dense search over thousands
of chunks genuinely returns near-misses in positions 6 through 20. To
demonstrate that here you would need a corpus large enough for top-k to be a
real filter. **The honest conclusion from this experiment is not "reranking is
useless" but "this benchmark is too small to measure reranking".** Reporting
the first would be as wrong as the fabricated table was.

### Security Test Suite: 15/15 passed (100%)

## How to Run

### 1. Setup

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/llm/rag-expert-assistant

python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
# source .venv/bin/activate

pip install uv
uv pip install -r requirements.txt
```

FlashRank downloads a ~4MB cross-encoder model the first time it runs and
caches it after. There is no GPU requirement anywhere in this project.

### 2. Run the parts that need no API key

Start here. These 32 checks exercise the chunking, the id derivation, the
injection defence and the PII filters without a single network call.

```bash
python run.py test        # 32 tests
python run.py security    # injection + PII suite, prints its results
```

`run.py` works the same on Windows, macOS and Linux. The `scripts/*.sh` files
still exist for anyone who prefers them, but they need bash.

### 3. Add an API key for the parts that call a model

```bash
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env and replace the placeholder with your key
```

Get a free key at https://aistudio.google.com/app/apikey. Only one key is
needed for the whole project: the same `GOOGLE_API_KEY` covers embeddings and
generation, and FlashRank runs locally.

Copying `.env.example` without editing it is not enough. It ships
`GOOGLE_API_KEY=your-google-api-key-here`, and a placeholder is a perfectly
truthy string, so an "is the key set?" check passes and the run fails much
later inside an HTTP 400. `run.py` rejects placeholder values up front.

### 4. Run the pipeline and the evaluations

```bash
python run.py pipeline    # index the corpus, answer one question with citations
python run.py eval        # RAGAS scores over the fixed evaluation set
python run.py ab          # measured naive vs optimized comparison
```

`python run.py ab` builds two complete pipelines, runs 10 questions through
each and scores every answer with RAGAS. It makes a few hundred API calls and
takes several minutes. Results are written to
`artifacts/results/ab_comparison.json`.

Watch the indexing line from `python run.py pipeline`. It prints the number of
unique chunks alongside the collection total and warns when they disagree,
which is how a duplicated index shows up as a message rather than as silently
worse retrieval.

### Running a module directly

`run.py` is a thin wrapper. Every step can be run on its own:

```bash
python -m src.rag_pipeline
python -m src.evaluate
python -m src.ab_comparison
python -m src.security.sanitizer
pytest tests/ src/security/test_security.py -v
```

Note the test path. `pytest tests/` alone collects 17 tests; the other 15 are
in `src/security/test_security.py`.

## Project Structure

```
rag-expert-assistant/
├── configs/
│   └── base.yaml              # Pipeline configuration (chunk size, models, thresholds)
├── data/
│   └── sample_docs/           # 4 sample documents for the RAG pipeline
├── notebooks/
│   └── RAG_Expert_Assistant.ipynb  # Interactive walkthrough notebook
├── src/
│   ├── rag_pipeline.py        # Full RAG: load -> chunk -> embed -> retrieve -> rerank -> generate
│   ├── evaluate.py            # RAGAS evaluation (faithfulness, relevancy, precision, recall)
│   ├── ab_comparison.py       # Naive vs Optimized RAG configuration comparison
│   └── security/
│       ├── sanitizer.py       # PII detection, prompt injection defense, output filtering
│       └── test_security.py   # Security test suite (injection + PII tests)
├── tests/
│   └── test_rag.py            # Unit tests for pipeline, security, and evaluation
├── docs/
│   └── architecture.md        # RAG pipeline architecture documentation
├── scripts/                   # bash wrappers (run.py covers the same ground)
├── .env.example               # API key template (Google only)
├── requirements.txt           # Dependencies
├── run.py                     # Cross-platform entry point: test, security, pipeline, eval, ab
└── README.md
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector store | ChromaDB | Simple setup, persistent storage, good for prototyping |
| Embeddings | gemini-embedding-001 | Free tier in Gemini API, 768 dims |
| Chunking | 512 chars, 50 overlap | Preserves context at sentence boundaries |
| Reranking | FlashRank (local) | No API key needed, runs locally, fast |
| Evaluation | RAGAS framework | Industry standard, separates retrieval vs generation quality |
| Security | Regex PII + pattern blocking | Fast, no external deps, catches 90%+ of common threats |
| Generation | Gemini 3.5 Flash Lite | Fast, cost-effective Gemini model for grounded RAG responses |

## Experiment Log

> ⚠️ Illustrative targets showing the *shape* of a tuning progression, not
> measurements. See the A/B note above. Fill these in from a real
> `evaluate_rag()` run.

| # | Experiment | Faithfulness | Precision | Key Change |
|---|-----------|-------------|-----------|------------|
| 1 | Naive (1000 chunks, top-3) | — | — | Baseline |
| 2 | Smaller chunks (512, overlap 50) | — | — | Chunking |
| 3 | Add reranking | — | — | Cross-encoder rerank |
| 4 | Grounded system prompt | — | — | Prompt constraint |

## The bug that made retrieval return one document

Worth reading even if you skip the rest, because nothing failed and nothing
warned.

`Chroma.from_documents(..., persist_directory=...)` **appends** to an existing
collection — it does not replace it. So every re-run of the pipeline added
another full copy of every chunk. The store committed here had reached **54
rows for 9 distinct chunks**: six identical copies of everything.

Duplicates carry identical embeddings, so they score identically. A top-20
similarity search returned the same handful of chunks over and over, the
reranker faithfully ranked those duplicates, and the top 5 passed to the model
were **five copies of one chunk**:

| | returned | distinct |
|---|---|---|
| before | 5 | **1** |
| after | 5 | **5** |

A retrieval system that returns one document is not a retrieval system: the
other eight chunks were unreachable, the context window filled with the same
paragraph, and you paid for the tokens. Indexing is now idempotent — each chunk
gets a content-addressed SHA-256 id, so re-running upserts instead of appending
and the collection size stays equal to the number of distinct chunks.

**Generalise it:** any indexing step you run more than once needs an identity
story. "It got bigger" is not "it got better".

## Interview Guide

**Q: Why use RAG instead of fine-tuning?**
> RAG keeps the model general while grounding answers in up-to-date documents. Fine-tuning bakes knowledge into weights (expensive to update, risks catastrophic forgetting). RAG lets you update the knowledge base by adding documents, no retraining needed.

**Q: How do you evaluate RAG quality?**
> RAGAS framework with 4 metrics: faithfulness (does the answer stick to context?), answer relevancy (does it address the question?), context precision (are retrieved chunks relevant?), context recall (did retrieval find all relevant info?). Each metric isolates a different failure mode.

**Q: Why rerank instead of just increasing top-k?**
> Cosine similarity ranks on embedding distance alone, which is lexically blunt; a cross-encoder scores the query and passage *together* and catches relevance that distance misses. So retrieve top-20 cheaply with dense search, then rerank to top-5. On **this** repo's corpus the measured gain is **exactly zero** (context precision 0.950 in both arms, three runs), because 9 chunks means "top-20" already returns the whole corpus and the reranker has nothing to rescue. The earlier "+17%" came from `random.uniform`, not an experiment. I would expect a real gain at corpus scale, and I would want to measure it there before quoting a number.

**Q: How do you prevent prompt injection?**
> Defense in depth: (1) Input sanitization strips known injection patterns, (2) System prompt constrains the model to context-only answers, (3) Output PII filter redacts any leaked personal data. We test with a 5-case injection suite.

**Q: What's the biggest limitation of this system?**
> That the demo corpus is too small to evaluate it. 4 documents and 3,396 characters means retrieval is trivially perfect (context recall measures 1.000) and the reranking and chunking work cannot show any benefit -- the measured A/B has the optimized pipeline *losing* on faithfulness and tying on precision. The fix is a bigger evaluation corpus and a question set with genuine near-misses, not another retrieval trick. An earlier version of this README named context recall (0.81) as the weakest metric; that number was not reproducible, and recall is in fact the strongest.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
