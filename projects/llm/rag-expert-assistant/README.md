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

Measured on 2026-11-13 with `python run.py eval --runs 3 --out
artifacts/results/eval_runs.json`, judged by `gemini-3.5-flash-lite` through
the ragas 0.4 collections API (`src/ragas_scoring.py`). Every per-question
score is in [`docs/results/2026-11-13/eval_runs.json`](docs/results/2026-11-13/eval_runs.json). RAGAS is an LLM-judged metric, so the spread across
runs is part of the result and is reported here rather than hidden behind a
single decimal.

| Metric | Run 1 | Run 2 | Run 3 | Status |
|--------|-------|-------|-------|--------|
| Faithfulness | 0.938 | 1.000 | 0.854 | UNSTABLE (one run below 0.85) |
| Answer Relevancy | 0.905 | 0.901 | 0.903 | PASS |
| Context Precision | 1.000 | 1.000 | 1.000 | PASS |
| Context Recall | 1.000 | 1.000 | 1.000 | PASS |

The variance sits in faithfulness. On identical input, one question scored
1.00, 1.00, then 0.67, and another 0.75, 1.00, then 0.75. These scores replace
an earlier table measured with the legacy ragas API (`ragas.metrics` +
`evaluate()`, deprecated in 0.4 and removed in 1.0). The collections metrics
score differently, so the two tables are not comparable number for number.

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

Measured on 2026-11-13 over three runs (`python run.py ab --out ...`), 10
questions per arm, judged by `gemini-3.5-flash-lite` through the ragas 0.4
collections API, with zero failed judge calls. The raw files are in
[`docs/results/2026-11-13/`](docs/results/2026-11-13/).

| Metric | Naive | Optimized | Delta | Runs agreeing |
|--------|-------|-----------|-------|---------------|
| Faithfulness | 0.975 / 0.967 / 1.000 | 0.917 / 0.947 / 0.942 | **-0.058 / -0.020 / -0.058** | WORSE in 3/3 |
| Answer Relevancy | 0.942 / 0.932 / 0.931 | 0.927 / 0.934 / 0.940 | -0.015 / +0.002 / +0.008 | no clear change |
| Context Precision | 0.950 / 1.000 / 1.000 | 1.000 / 0.975 / 1.000 | +0.050 / -0.025 / 0.000 | no clear change |
| Context Recall | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | SAME in 3/3 |

Optimized loses faithfulness in every run and moves nothing else
consistently. It is about **1.2x slower** (246 s against 204 s per arm, mean of
three runs, judging included). The old fabricated table claimed it won all
four, with "+24% faithfulness" and "+17% context precision". An earlier real
run with the legacy ragas API also reported 2 wins of 4; the collections
metrics do not reproduce those small wins, which is itself a reminder that
deltas of a few hundredths are inside the judge's noise.

**Why the optimizations do nothing here, and it is not because they are bad
ideas.** The demo corpus is 4 documents totalling **3,396 characters** -- about
one page of text. That splits into **5 chunks** under the naive config and
**9 chunks** under the optimized one. The optimized retriever asks for the top
20 candidates from a store containing 9, so stage one returns *the entire
corpus* and the reranker's job reduces to discarding 4 of 9 chunks. Two-stage
retrieval cannot improve precision when the first stage already returns
everything, which is why context precision only wobbles by a few hundredths
between runs (+0.050, -0.025, 0.000): that movement is the judge, not the
retriever.

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
python run.py test        # full test suite, no API key
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

Note the test path. `pytest tests/` alone skips the 15 security tests, which
live in `src/security/test_security.py`. Every test runs without an API key.

## Project Structure

```
rag-expert-assistant/
├── configs/
│   └── base.yaml              # Pipeline configuration (embedding model + dimensions are read at runtime; other sections are reference values)
├── data/
│   └── sample_docs/           # 4 sample documents for the RAG pipeline
├── notebooks/
│   └── RAG_Expert_Assistant.ipynb  # Interactive walkthrough notebook
├── src/
│   ├── rag_pipeline.py        # Full RAG: load -> chunk -> embed -> retrieve -> rerank -> generate, security on the query path
│   ├── reranking.py           # FlashRank reranker + two-stage retriever (langchain-core)
│   ├── evaluate.py            # RAGAS evaluation (faithfulness, relevancy, precision, recall)
│   ├── ragas_scoring.py       # Shared RAGAS scorer (ragas 0.4 collections API, async Gemini judge)
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
| Embeddings | gemini-embedding-001 | Free tier in Gemini API, 3072 dims (set by `embedding.dimensions` in `configs/base.yaml`) |
| Chunking | 512 chars, 50 overlap | Preserves context at sentence boundaries |
| Reranking | FlashRank `ms-marco-MultiBERT-L-12` (local) | No API key needed, runs locally; the model is pinned in `configs/base.yaml` because every published score was measured with it |
| Evaluation | RAGAS framework | Industry standard, separates retrieval vs generation quality |
| Security | Regex PII + pattern blocking, run on every query | Fast, no external deps. Catches the formats its patterns describe (4 PII types, 4 injection phrases) and nothing else; no detection rate is claimed because none was measured |
| Generation | Gemini 3.5 Flash Lite | Fast, cost-effective Gemini model for grounded RAG responses |

## Experiment Log

> ⚠️ Illustrative targets showing the *shape* of a tuning progression, not
> measurements. See the A/B note above. Fill these in from a real
> `evaluate_rag()` run.

| # | Experiment | Faithfulness | Precision | Key Change |
|---|-----------|-------------|-----------|------------|
| 1 | Naive (1000 chunks, top-3) | - | - | Baseline |
| 2 | Smaller chunks (512, overlap 50) | - | - | Chunking |
| 3 | Add reranking | - | - | Cross-encoder rerank |
| 4 | Grounded system prompt | - | - | Prompt constraint |

## The bug that made retrieval return one document

Worth reading even if you skip the rest, because nothing failed and nothing
warned.

`Chroma.from_documents(..., persist_directory=...)` **appends** to an existing
collection - it does not replace it. So every re-run of the pipeline added
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
paragraph, and you paid for the tokens. Indexing is now idempotent - each chunk
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
