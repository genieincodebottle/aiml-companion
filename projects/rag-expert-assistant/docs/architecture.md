# RAG Pipeline Architecture

This document describes the end-to-end architecture of the RAG Expert Assistant, covering each stage from document ingestion through evaluation.

## Pipeline Overview

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

## Stage Details

### 1. Ingest

Documents are loaded from `data/sample_docs/` using LangChain's `DirectoryLoader`. The loader supports `.txt` and `.md` files. In production, this stage would also handle PDF, HTML, and other formats via appropriate loaders.

- **Entry point**: `src/rag_pipeline.py :: load_documents()`
- **Input**: Raw documents (TXT, MD, PDF)
- **Output**: List of LangChain `Document` objects with metadata

### 2. Chunk

Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter`. The splitter respects semantic boundaries (paragraphs, sentences) to preserve context.

- **Entry point**: `src/rag_pipeline.py :: chunk_documents()`
- **Configuration** (from `configs/base.yaml`):
  - `chunk_size`: 512 characters
  - `chunk_overlap`: 50 characters
  - `separators`: `["\n\n", "\n", ". ", " "]`
- **Output**: List of chunked `Document` objects

### 3. Embed

Each chunk is embedded into a 1536-dimensional vector using OpenAI's `text-embedding-3-small` model. Embeddings are stored in ChromaDB for persistent vector search.

- **Entry point**: `src/rag_pipeline.py :: build_vectorstore()`
- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Vector store**: ChromaDB with persistent storage
- **Output**: Populated ChromaDB collection

### 4. Retrieve

Given a user query, the retriever performs dense similarity search against ChromaDB to fetch the top-k candidate chunks. The initial retrieval casts a wide net (top-20) to maximize recall before reranking.

- **Entry point**: `src/rag_pipeline.py :: build_retriever()`
- **Search type**: Cosine similarity
- **top_k_retrieval**: 20 candidates
- **Output**: List of candidate documents with similarity scores

### 5. Rerank

A cross-encoder reranker (Cohere rerank-v3.5) re-scores the top-20 candidates and selects the top-5 most relevant chunks. Reranking captures semantic relevance that cosine similarity misses, providing a +17% improvement in context precision.

- **Entry point**: `src/rag_pipeline.py :: build_retriever()` (with `use_reranking=True`)
- **Model**: `cohere-rerank-v3.5`
- **top_k_rerank**: 5 (from 20 candidates)
- **Output**: Top-5 reranked documents

### 6. Generate

The LLM generates a grounded response using the reranked context. The system prompt enforces citation rules, confidence ratings, and context-only answering to minimize hallucination.

- **Entry point**: `src/rag_pipeline.py :: build_rag_chain()`
- **Model**: `gpt-4o-mini` (temperature=0)
- **Prompt strategy**: Grounded system prompt with `[Source N]` citation format
- **Security**: Input sanitization before query, output PII filtering after generation
- **Output**: Answer with citations and confidence rating

### 7. Evaluate

The RAGAS framework evaluates pipeline quality across four independent metrics, each isolating a different failure mode. An A/B comparison framework measures the impact of each optimization.

- **Entry points**:
  - `src/evaluate.py :: run_evaluation()` -- RAGAS metrics
  - `src/ab_comparison.py :: run_ab_comparison()` -- Naive vs Optimized
- **Metrics**:
  - **Faithfulness**: Does the answer stick to the retrieved context?
  - **Answer Relevancy**: Does the answer address the question?
  - **Context Precision**: Are retrieved chunks relevant to the question?
  - **Context Recall**: Did retrieval find all relevant information?
- **Output**: Evaluation reports in `artifacts/results/`

## Security Layer

The security module (`src/security/`) provides defense-in-depth across the pipeline:

1. **Input sanitization** (`sanitize_input`): Blocks known prompt injection patterns before they reach the LLM
2. **PII detection** (`detect_pii`): Scans input and context for email, phone, SSN, and credit card patterns
3. **Output filtering** (`filter_output_pii`): Redacts any PII that appears in LLM responses before returning to the user

## Project Layout

```
rag-expert-assistant/
├── configs/base.yaml            # Pipeline configuration (chunk size, models, thresholds)
├── data/sample_docs/            # Source documents for the RAG pipeline
├── notebooks/                   # Interactive Jupyter notebook walkthrough
├── src/
│   ├── rag_pipeline.py          # Full RAG: load -> chunk -> embed -> retrieve -> rerank -> generate
│   ├── evaluate.py              # RAGAS evaluation (faithfulness, relevancy, precision, recall)
│   ├── ab_comparison.py         # Naive vs Optimized RAG configuration comparison
│   └── security/
│       ├── sanitizer.py         # PII detection, prompt injection defense, output filtering
│       └── test_security.py     # Security test suite (injection + PII tests)
├── tests/
│   └── test_rag.py              # Unit tests for pipeline, security, and evaluation
├── artifacts/results/           # Generated evaluation and security reports
├── docs/architecture.md         # This file
├── scripts/                     # Shell scripts for running pipeline and evaluation
├── .env.example                 # API key template (OpenAI + Cohere)
├── Makefile                     # Build targets (run, evaluate, test, etc.)
├── requirements.txt             # Pinned dependencies
└── README.md                    # Project overview and quickstart
```
