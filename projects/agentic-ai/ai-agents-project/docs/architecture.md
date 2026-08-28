# Architecture: Multi-Agent Research System

This document describes the system architecture of the Multi-Agent Research pipeline,
including the 4-agent flow, shared state design, and guardrails layer.

## Overview

The system uses **LangGraph StateGraph** to orchestrate four specialized agents in a
sequential pipeline. Each agent handles a distinct phase of the research process,
communicating through a shared typed state dictionary. A guardrails layer wraps the
pipeline to enforce PII scrubbing, URL validation, and token budget constraints.

## Pipeline Flow

The agents execute in a fixed sequential order. Each agent reads from and writes to
the shared `ResearchState`, enabling downstream agents to build on upstream outputs.

```
                         User Query
                             |
                             v
            +-------------------------------+
            |       Guardrails Layer        |
            |  PII Scrub | URL Validate     |
            |  Budget Enforcement (50K cap) |
            +-------------------------------+
                             |
                             v
  +-------------+    +-------------+    +-------------+    +--------------+
  |             |    |             |    |             |    |              |
  | Researcher  |--->|  Analyst    |--->|   Writer    |--->| Fact-Checker |
  |             |    |             |    |             |    |              |
  +------+------+    +------+------+    +------+------+    +------+-------+
         |                  |                  |                  |
         |  sources[]       |  key_claims[]    |  draft           |  final_report
         |  (web search     |  (structured     |  (formatted      |  (verified &
         |   results)       |   claims with    |   report from    |   revised with
         |                  |   confidence)    |   claims only)   |   [NEEDS CITATION]
         |                  |                  |                  |   markers)
         +------------------+------------------+------------------+
                                    |
                                    v
                          Shared ResearchState
```

## Agent Responsibilities

### 1. Researcher

- **Input:** `query` (user's research question)
- **Output:** `sources[]` (list of {title, url, snippet, date})
- **Tool:** Tavily Search API (returns structured JSON, no HTML parsing needed)
- **Budget check:** Skips execution if token budget is exhausted
- **Error handling:** Returns empty sources list with error logged to `state.errors`

### 2. Analyst

- **Input:** `sources[]` from Researcher
- **Output:** `key_claims[]` (list of {claim, source_idx, confidence})
- **Task:** Prompts the LLM to extract 5-8 key claims, each tied to a source index
  and rated by confidence (high/medium/low)
- **Budget check:** Skips execution if token budget is exhausted
- **Error handling:** Returns empty claims list with error logged

### 3. Writer

- **Input:** `key_claims[]` from Analyst, `sources[]` from Researcher
- **Output:** `draft` (structured report with sections: Introduction, Key Findings,
  Analysis, Sources)
- **Constraint:** The writer is prompted to use ONLY the provided claims and sources.
  This prevents hallucinated citations.
- **Budget check:** Skips execution if token budget is exhausted

### 4. Fact-Checker

- **Input:** `draft` from Writer, `sources[]` from Researcher
- **Output:** `final_report`, `fact_check[]` (verification metadata)
- **Process:**
  1. Verifies each claim in the draft against source snippets
  2. Flags unsupported claims with `[NEEDS CITATION]` markers
  3. Scrubs PII from the final output (emails, phone numbers, SSNs)
  4. Validates source URLs via HEAD requests
- **Budget check:** Returns draft as-is if budget is exhausted

## Shared State Design

All agents read from and write to a single `ResearchState` TypedDict:

```python
class ResearchState(TypedDict):
    query: str              # User's research question
    sources: list[dict]     # Retrieved sources (from Researcher)
    key_claims: list[dict]  # Extracted claims (from Analyst)
    draft: str              # Written report (from Writer)
    fact_check: list[dict]  # Verification results (from Fact-Checker)
    final_report: str       # Polished output (from Fact-Checker)
    token_count: int        # Cumulative token tracking (all agents)
    errors: list[str]       # Error log (all agents)
```

Key design decisions for shared state:

- **Cumulative token tracking:** Each agent adds its token usage to `token_count`,
  enabling global budget enforcement without per-agent limits.
- **Append-only error log:** Agents append to `errors` rather than overwriting,
  preserving the full error history across the pipeline.
- **Graceful degradation:** When the budget is exceeded, agents skip their LLM call
  and pass through existing state, producing the best report possible.

## Guardrails Layer

The guardrails module (`src/guardrails.py`) provides three safety mechanisms:

### PII Detection and Scrubbing

- Compiled regex patterns for email, phone, and SSN
- `detect_pii(text)` returns matched PII by type
- `scrub_pii(text)` replaces PII with `[REDACTED_TYPE]` tokens
- Applied by the Fact-Checker before producing the final report

### URL Validation

- HEAD request with 5-second timeout to verify source URLs are reachable
- Returns `True` if status code < 400
- Applied by the Fact-Checker to audit source quality

### Token Budget Enforcement

- Global 50,000-token hard cap stored in `TOKEN_BUDGET`
- `check_budget(current_tokens)` returns `False` when budget is exceeded
- Every agent checks budget before making LLM calls
- Prevents runaway API costs in production

## Graph Construction

The LangGraph `StateGraph` is built in `src/agents.py`:

```
Entry Point --> researcher --> analyst --> writer --> fact_checker --> END
```

All edges are deterministic (no conditional branching in the current design).
This keeps the pipeline debuggable and predictable. Conditional edges (e.g., skipping
fact-checking for low-stakes queries) can be added as future optimizations.

## Evaluation Architecture

The evaluation framework (`evaluation/`) is kept as a separate top-level module
because it operates independently of the main pipeline:

- **`run_eval.py`:** Runs single-agent vs multi-agent comparison across a test set
  of 10 research questions. Uses LLM-as-judge scoring on accuracy, completeness,
  and citation quality (0-3 scale).
- **`judge_prompt.py`:** Contains the scoring prompts used by the LLM judge.

The evaluation module uses the OpenAI SDK directly (not LangGraph) to keep the
baseline comparison fair and the evaluation logic decoupled from the agent framework.
