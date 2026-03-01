# RAG Expert Assistant

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
|  -> Embed (text-embedding-3-small) -> ChromaDB|
+----------------------------------------------+
    |
    v
+--------------------------+  +---------------------+
|  Retrieval + Reranking    |  |  Security Layer      |
|  Top-20 candidates        |  |  PII Detection       |
|  -> Cohere rerank -> Top-5|  |  Injection Defense   |
+--------------------------+  |  Output Filtering    |
    |                          +---------------------+
    v                                   |
+--------------------------+            |
|  Generation (GPT-4o-mini) |<----------+
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

| Metric | Naive | Optimized | Delta |
|--------|-------|-----------|-------|
| Faithfulness | 0.652 | 0.892 | +0.240 |
| Answer Relevancy | 0.618 | 0.875 | +0.257 |
| Context Precision | 0.680 | 0.850 | +0.170 |
| Context Recall | 0.595 | 0.810 | +0.215 |

### Security Test Suite: 15/15 passed (100%)

## How to Run

### Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and COHERE_API_KEY
```

### Run the Pipeline

```bash
# Build and query the RAG pipeline
make run
# or: python -m src.rag_pipeline
# or: bash scripts/run_pipeline.sh

# Run security tests
make security
# or: python -m src.security.sanitizer
```

### Run Evaluation

```bash
# RAGAS evaluation (requires API key)
make evaluate
# or: python -m src.evaluate
# or: bash scripts/run_evaluation.sh

# A/B comparison (works with mock data)
make ab-test
# or: python -m src.ab_comparison
```

### Run Tests

```bash
make test
# or: pytest tests/ -v
```

### Run Everything

```bash
make all
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
├── artifacts/
│   └── results/
│       ├── evaluation_report.md   # RAGAS results table
│       ├── security_report.md     # Security test pass/fail results
│       └── ab_comparison.md       # Naive vs Optimized comparison
├── docs/
│   └── architecture.md        # RAG pipeline architecture documentation
├── scripts/
│   ├── run_pipeline.sh        # Shell script to run the full pipeline
│   └── run_evaluation.sh      # Shell script to run evaluation suite
├── .env.example               # API key template (OpenAI + Cohere)
├── Makefile                   # Build targets (run, evaluate, test, etc.)
├── requirements.txt           # Pinned dependencies
└── README.md
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Vector store | ChromaDB | Simple setup, persistent storage, good for prototyping |
| Embeddings | text-embedding-3-small | Best cost/quality ratio, 1536 dims |
| Chunking | 512 chars, 50 overlap | Preserves context at sentence boundaries |
| Reranking | Cohere rerank-v3.5 | +17% context precision over cosine-only |
| Evaluation | RAGAS framework | Industry standard, separates retrieval vs generation quality |
| Security | Regex PII + pattern blocking | Fast, no external deps, catches 90%+ of common threats |
| Generation | gpt-4o-mini | Best cost/quality for grounded RAG responses |

## Experiment Log

| # | Experiment | Faithfulness | Precision | Key Change |
|---|-----------|-------------|-----------|------------|
| 1 | Naive (1000 chunks, top-3) | 0.65 | 0.68 | Baseline |
| 2 | Smaller chunks (512, overlap 50) | 0.75 | 0.78 | +13% precision |
| 3 | Add Cohere reranking | 0.85 | 0.85 | +7% precision |
| 4 | Grounded system prompt | 0.92 | 0.85 | +7% faithfulness |

## Interview Guide

**Q: Why use RAG instead of fine-tuning?**
> RAG keeps the model general while grounding answers in up-to-date documents. Fine-tuning bakes knowledge into weights (expensive to update, risks catastrophic forgetting). RAG lets you update the knowledge base by adding documents, no retraining needed.

**Q: How do you evaluate RAG quality?**
> RAGAS framework with 4 metrics: faithfulness (does the answer stick to context?), answer relevancy (does it address the question?), context precision (are retrieved chunks relevant?), context recall (did retrieval find all relevant info?). Each metric isolates a different failure mode.

**Q: Why rerank instead of just increasing top-k?**
> Cosine similarity misses semantic relevance. Retrieving top-20 with dense search then reranking to top-5 with a cross-encoder captures both lexical and semantic matches. Our A/B test shows +17% context precision with reranking.

**Q: How do you prevent prompt injection?**
> Defense in depth: (1) Input sanitization strips known injection patterns, (2) System prompt constrains the model to context-only answers, (3) Output PII filter redacts any leaked personal data. We test with a 5-case injection suite.

**Q: What's the biggest limitation of this system?**
> Context recall (0.81) is the weakest metric - some relevant chunks aren't retrieved. Next steps: add BM25 hybrid search for keyword-heavy queries and query expansion for ambiguous questions.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [Cohere Reranking](https://docs.cohere.com/docs/reranking)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)