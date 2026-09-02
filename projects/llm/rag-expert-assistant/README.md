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
|  Generation (Gemini 2.5   |<----------+
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

| Metric | Score | Status |
|--------|-------|--------|
| Faithfulness | 0.920 | PASS |
| Answer Relevancy | 0.875 | PASS |
| Context Precision | 0.850 | PASS |
| Context Recall | 0.810 | NEEDS WORK |

### Naive vs Optimized RAG (A/B Comparison)

> ⚠️ **Not measured. This table is the report LAYOUT, not a result.**
> `src/ab_comparison.py` is a skeleton: `evaluate_rag()` never ran a pipeline.
> It drew scores from `random.uniform` around a hardcoded base of **0.65 for
> "Naive"** and **0.88 for "Optimized"**, so the optimized config won by ~0.23
> every single run — not because it retrieves better, but because 0.88 is a
> bigger number than 0.65. The conclusion was written before the experiment.
>
> The function now **raises** unless you pass `--allow-mock`, and the mock run
> prints a banner on every invocation. Wire it to the real pipeline (the
> docstring lists the four steps) and replace these figures with measurements.

| Metric | Naive | Optimized | Delta |
|--------|-------|-----------|-------|
| Faithfulness | *not measured* | *not measured* | — |
| Answer Relevancy | *not measured* | *not measured* | — |
| Context Precision | *not measured* | *not measured* | — |
| Context Recall | *not measured* | *not measured* | — |

**The lesson this replaced a fake table to teach:** A/B numbers you did not
measure are worse than no numbers. They travel — into a README, then a slide,
then a decision — and nothing about their formatting distinguishes them from
real ones. If an evaluation harness is a stub, it must fail loudly rather than
return plausible floats.

### Security Test Suite: 15/15 passed (100%)

## How to Run

### 1. Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
# source .venv/bin/activate

# Install uv (fast package installer, one-time setup)
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Copy the example and add your Google API key
cp .env.example .env
# Edit .env with your key from https://aistudio.google.com/app/apikey
```

Only one API key needed - Google API key (free tier). No other keys required.

### 3. Run

```bash
# Run the RAG pipeline (ingest, chunk, embed, retrieve, generate)
python -m src.rag_pipeline

# Run evaluation (RAGAS metrics)
python -m src.evaluate

# Run A/B comparison (naive vs optimized)
python -m src.ab_comparison

# Run security tests (injection + PII)
python -m src.security.sanitizer

# Run unit tests
pytest tests/ -v
```

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
├── .env.example               # API key template (Google only)
├── requirements.txt           # Dependencies
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
> Cosine similarity ranks on embedding distance alone, which is lexically blunt; a cross-encoder scores the query and passage *together* and catches relevance that distance misses. So retrieve top-20 cheaply with dense search, then rerank to top-5. I am deliberately **not** quoting a number for the gain here: the A/B harness in this repo is still a stub, and the "+17%" this answer used to cite came from `random.uniform`, not an experiment. Measuring it is the next task.

**Q: How do you prevent prompt injection?**
> Defense in depth: (1) Input sanitization strips known injection patterns, (2) System prompt constrains the model to context-only answers, (3) Output PII filter redacts any leaked personal data. We test with a 5-case injection suite.

**Q: What's the biggest limitation of this system?**
> Context recall (0.81) is the weakest metric - some relevant chunks aren't retrieved. Next steps: add BM25 hybrid search for keyword-heavy queries and query expansion for ambiguous questions.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
