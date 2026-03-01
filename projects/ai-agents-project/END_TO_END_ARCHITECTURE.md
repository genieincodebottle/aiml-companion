# Multi-Agent Research System - End-to-End Architecture

> Comprehensive technical reference for the 7-agent orchestrated research pipeline (8 graph nodes)
> with quality-gated routing, parallel fan-out, iterative refinement, and production guardrails.
>
> **100% free to run.** Uses Google Gemini free tier + Tavily free tier. No credit card or payment needed.

---

## Table of Contents

1. [High-Level Architecture Diagram](#1-high-level-architecture-diagram)
2. [Backend Module Overview](#2-backend-module-overview)
3. [LLM and Tool Integration](#3-llm-and-tool-integration)
4. [Shared State Schema](#4-shared-state-schema)
5. [Complete Research Query Flow - Happy Path](#5-complete-research-query-flow---happy-path)
6. [Complete Research Query Flow - Quality Gate Retry](#6-complete-research-query-flow---quality-gate-retry)
7. [Complete Research Query Flow - Reviewer Refinement Loop](#7-complete-research-query-flow---reviewer-refinement-loop)
8. [Sequence Diagrams](#8-sequence-diagrams)
9. [Database Schema](#9-database-schema)
10. [Performance Metrics](#10-performance-metrics)
11. [Security and Privacy](#11-security-and-privacy)
12. [Configuration Guide](#12-configuration-guide)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Testing Architecture](#14-testing-architecture)
15. [Complete Flow Summary](#15-complete-flow-summary)

---

## 1. High-Level Architecture Diagram

The system orchestrates 7 specialized agent functions across 8 graph nodes through a LangGraph StateGraph. The pipeline
features three advanced patterns: parallel fan-out via `Send()`, quality-gated conditional
routing, and bounded iterative refinement.

```
                              +------------------+
                              |   User Query     |
                              | (Streamlit UI /  |
                              |  CLI / Python)   |
                              +--------+---------+
                                       |
                                       v
                    +--------------------------------------+
                    |         GUARDRAILS LAYER             |
                    |                                      |
                    |  [PII Detection]  Regex scan for     |
                    |                   email/phone/SSN    |
                    |                                      |
                    |  [Token Budget]   50K cap with       |
                    |                   graceful degrade   |
                    |                                      |
                    |  [Rate Limiter]   30 RPM for         |
                    |                   Gemini free tier   |
                    |                                      |
                    |  [Injection Det]  Regex patterns for  |
                    |                   prompt injection    |
                    +--------------------------------------+
                                       |
                                       v
                              +--------+---------+
                              |    PLANNER       |
                              |                  |
                              | Decompose query  |
                              | into 1-3 focused |
                              | sub-topics via   |
                              | structured_output|
                              +--------+---------+
                                       |
                          Send() parallel fan-out
                       +-------+-------+-------+
                       |       |       |       |
                       v       v       v       v
                   +------+ +------+ +------+
                   |RSRCHR| |RSRCHR| |RSRCHR|  (1 per sub-topic)
                   |  #1  | |  #2  | |  #3  |
                   |      | |      | |      |  Each uses:
                   |Tavily| |Tavily| |Tavily|  - Tavily Search
                   |+Wiki | |+Wiki | |+Wiki |  - Wikipedia API
                   +--+---+ +--+---+ +--+---+  - BeautifulSoup
                      |        |        |
                      +--------+--------+
                               |
                     operator.add merges
                     sources[] into state
                               |
                               v
                     +---------+---------+
                     |   QUALITY GATE    |
                     |                   |
                     | Pure Python -     |
                     | no LLM call       |
                     |                   |
                     | domain_trust(0.6) |
                     | + snippet(0.4)    |
                     | = combined score  |
                     +---------+---------+
                               |
                    +----------+----------+
                    |                     |
              score < 0.4           score >= 0.4
              (FAIL)                (PASS)
                    |                     |
                    v                     v
           +-------+-------+    +--------+--------+
           | RETRY         |    |    ANALYST       |
           | RESEARCHER    |    |                  |
           |               |    | Extract 5-8      |
           | Broadened     |    | claims with      |
           | query with    |    | evidence +       |
           | Wikipedia     |    | confidence via   |
           +-------+-------+    | structured_output|
                   |            +--------+--------+
                   |                     |
                   +------> analyst      |
                                         v
                              +----------+---------+
                              |    SYNTHESIZER     |
                              |                    |
                              | Cross-reference    |
                              | claims, detect     |
                              | conflicts, rank    |
                              | source reliability |
                              +----------+---------+
                                         |
                                         v
                              +----------+---------+
                              |      WRITER        |
                              |                    |
                              | Versioned drafts   |
                              | with citations     |
                              | PII scrubbed       |
                        +---->| from output        |
                        |     +----------+---------+
                        |                |
                        |                v
                        |     +----------+---------+
                        |     |     REVIEWER       |
                        |     |                    |
                        |     | Score 1-10 via     |
                        |     | structured_output  |
                        |     | Flag issues +      |
                        |     | suggestions        |
                        |     +----------+---------+
                        |                |
                        |     +----------+----------+
                        |     |                     |
                        | score < 7             score >= 7
                        | AND rev < 2           OR rev >= 2
                        |     |                     |
                        +-----+                     v
                    (refinement              +-----------+
                     loop, max 2)            |    END    |
                                             |           |
                                             | final_    |
                                             | report    |
                                             | set       |
                                             +-----------+
```

### Graph Topology Summary

```
NODES (8):
  planner -> researcher (via Send, parallel) -> quality_gate
  quality_gate -> [conditional] -> analyst OR retry_researcher
  retry_researcher -> analyst
  analyst -> synthesizer -> writer -> reviewer
  reviewer -> [conditional] -> writer (loop) OR END

EDGES:
  planner          --> researcher         (Send fan-out, 1-3 parallel instances)
  researcher       --> quality_gate       (deterministic)
  quality_gate     --> analyst            (conditional: quality_passed=True)
  quality_gate     --> retry_researcher   (conditional: quality_passed=False)
  retry_researcher --> analyst            (deterministic, bypasses quality re-check)
  analyst          --> synthesizer        (deterministic)
  synthesizer      --> writer             (deterministic)
  writer           --> reviewer           (deterministic)
  reviewer         --> END                (conditional: review.passed=True OR max revisions)
  reviewer         --> writer             (conditional: review.passed=False AND rev < max)
```

---

## 2. Backend Module Overview

### Directory Structure

```
ai-agents-project/
├── src/
│   ├── agents/                          # 7 agent functions (8 graph nodes) + graph wiring
│   │   ├── __init__.py                  # Re-exports build_graph, run_pipeline
│   │   ├── planner.py                   # Query decomposition (structured output)
│   │   ├── researcher.py                # Multi-tool parallel research
│   │   ├── quality_gate.py              # Pure Python source scoring (no LLM)
│   │   ├── analyst.py                   # Claim extraction with evidence linking
│   │   ├── synthesizer.py               # Cross-source conflict detection
│   │   ├── writer.py                    # Versioned report generation + PII scrub
│   │   ├── reviewer.py                  # Draft review + scoring (structured output)
│   │   └── graph.py                     # LangGraph StateGraph wiring + routing
│   ├── tools/                           # Research tools (Tavily, Wikipedia, scraper)
│   │   ├── __init__.py
│   │   ├── search.py                    # Tavily web search wrapper
│   │   ├── wikipedia.py                 # Wikipedia REST API search
│   │   ├── scraper.py                   # BeautifulSoup HTML text extraction
│   │   └── tool_selector.py             # Query-type to tool routing
│   ├── cache/
│   │   ├── __init__.py
│   │   └── research_cache.py            # SQLite source/query cache (24h TTL)
│   ├── models/
│   │   ├── __init__.py
│   │   └── state.py                     # ResearchState TypedDict + Pydantic schemas
│   ├── config.py                        # YAML config loader with defaults
│   └── guardrails.py                    # PII, URL validation, budget, rate limiter, prompt injection
├── configs/
│   └── base.yaml                        # Agent registry + pipeline parameters
├── evaluation/
│   ├── __init__.py
│   ├── run_eval.py                      # Single vs multi-agent comparison framework
│   └── judge_prompt.py                  # LLM-as-judge scoring prompts
├── tests/                               # 110 tests across 8 files
│   ├── test_e2e_pipeline.py             # Full pipeline (37 tests, mocked LLM)
│   ├── test_ui_smoke.py                 # Streamlit UI imports (26 tests)
│   ├── test_guardrails.py               # Guardrails unit tests (14 tests)
│   ├── test_graph.py                    # Graph routing tests (9 tests)
│   ├── test_quality_gate.py             # Quality scoring tests (8 tests)
│   ├── test_state.py                    # State schema tests (6 tests)
│   ├── test_cache.py                    # Cache operation tests (5 tests)
│   └── test_tools.py                    # Tool selector tests (5 tests)
├── artifacts/results/
│   ├── sample_report.md                 # Example generated research report
│   ├── evaluation_results.md            # Single vs multi-agent comparison table
│   └── cost_analysis.md                 # Token usage breakdown
├── scripts/
│   ├── initialize_cache.py              # Seed SQLite cache with sample queries
│   ├── cleanup_data.py                  # Reset cache and clear __pycache__
│   ├── run_agents.sh                    # Shell wrapper to run pipeline
│   └── run_evaluation.sh               # Shell wrapper to run evaluation
├── app.py                               # Streamlit UI (graph viz, trace, streaming)
├── END_TO_END_ARCHITECTURE.md           # This document
└── README.md                            # Setup, testing guide, interview prep
```

### Module-by-Module Reference

---

#### `src/agents/graph.py` - LangGraph Pipeline Wiring

**Purpose:** Builds and compiles the 7-agent research pipeline (8 graph nodes) as a LangGraph StateGraph.
Wires all nodes, edges, conditional routing, and the Send() parallel fan-out.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `build_graph()` | Function | Creates the StateGraph, adds all nodes, wires edges and conditionals, returns compiled graph |
| `run_pipeline(query)` | Function | Convenience wrapper - builds graph, creates default state, invokes pipeline, returns final state |
| `route_to_researchers(state)` | Routing fn | Creates `Send("researcher", ...)` objects, one per sub-topic, for parallel fan-out |
| `route_after_quality(state)` | Routing fn | Returns "analyst" if quality passed, "retry_researcher" if first attempt failed, "analyst" if already retried |
| `route_after_review(state)` | Routing fn | Returns END if review passed or max revisions reached, "writer" otherwise |
| `retry_researcher(state)` | Node fn | Re-runs researcher with broadened query ("{query} comprehensive overview analysis") |

**Graph Construction Flow:**
```python
graph = StateGraph(ResearchState)

# Add nodes (7 functions, 8 nodes - retry_researcher reuses researcher)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("quality_gate", quality_gate)
graph.add_node("retry_researcher", retry_researcher)
graph.add_node("analyst", analyst)
graph.add_node("synthesizer", synthesizer)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

# Entry point
graph.set_entry_point("planner")

# Planner -> parallel researchers (Send fan-out)
graph.add_conditional_edges("planner", route_to_researchers, ["researcher"])

# Researchers merge -> quality gate
graph.add_edge("researcher", "quality_gate")

# Quality gate -> conditional: analyst or retry
graph.add_conditional_edges("quality_gate", route_after_quality,
    {"analyst": "analyst", "retry_researcher": "retry_researcher"})

# Retry researcher -> analyst (skip second quality check)
graph.add_edge("retry_researcher", "analyst")

# Linear chain: analyst -> synthesizer -> writer -> reviewer
graph.add_edge("analyst", "synthesizer")
graph.add_edge("synthesizer", "writer")
graph.add_edge("writer", "reviewer")

# Reviewer -> conditional: END or back to writer
graph.add_conditional_edges("reviewer", route_after_review,
    {"writer": "writer", END: END})

return graph.compile()
```

---

#### `src/agents/planner.py` - Query Decomposition

**Purpose:** Decomposes a research query into 1-3 focused sub-topics using
`with_structured_output(PlannerOutput)` for guaranteed JSON parsing.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `planner(state)` | Agent fn | Takes ResearchState, returns {sub_topics, research_plan, token_count, pipeline_trace} |

**LLM Call Details:**
- Model: `ChatGoogleGenerativeAI(model=get_model_name(), temperature=0)`
- Structured output: `llm.with_structured_output(PlannerOutput)` returns Pydantic object
- Output enforces: 1-3 sub_topics (list[str]), research_plan (str)
- Token estimate: ~500 tokens per call

**Error Handling:**
- Budget exceeded: Returns original query as single sub-topic, skips LLM call
- LLM error: Falls back to `[state["query"]]` as single sub-topic

---

#### `src/agents/researcher.py` - Multi-Tool Parallel Research

**Purpose:** Searches for sources using multiple tools (Tavily, Wikipedia, scraper).
Runs in parallel via LangGraph Send(), one instance per sub-topic.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `researcher(state)` | Agent fn | Takes {query, token_count}, returns {sources, search_queries_used, token_count, pipeline_trace} |

**Execution Flow:**
1. `select_tools(query)` determines which tools to use
2. For each selected tool, run the search and collect results
3. Deduplicate sources by URL
4. Return unique sources (merged into shared state via `operator.add`)

**Important:** This function does NOT make an LLM call. It only calls external search
APIs (Tavily, Wikipedia). Token estimate: ~300 per call (for state bookkeeping).

**Parallel Execution:**
When the planner produces N sub-topics, N instances of this function run in parallel.
Each instance receives a different sub-topic as its `query`. Results merge via
`Annotated[list[dict], operator.add]` on the `sources` field.

---

#### `src/agents/quality_gate.py` - Pure Python Source Scoring

**Purpose:** Scores source quality WITHOUT an LLM call using domain trust heuristics
and snippet quality analysis. Routes the pipeline: pass -> analyst, fail -> retry.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `quality_gate(state)` | Agent fn | Scores sources, returns {quality_score, quality_passed, source_ranking, pipeline_trace} |
| `_domain_score(url)` | Helper | Returns 0.0-1.0 trust score based on domain |
| `_snippet_score(snippet)` | Helper | Returns 0.0-1.0 quality score based on content heuristics |
| `DOMAIN_TRUST` | Dict | Domain -> trust score mapping |

**Scoring Formula:**
```
combined_score = 0.6 * domain_trust + 0.4 * snippet_quality
quality_score  = mean(top 5 combined scores)
quality_passed = quality_score >= threshold (default 0.4)
```

**Domain Trust Scores:**

| Domain | Score | Rationale |
|--------|-------|-----------|
| arxiv.org | 0.95 | Peer-reviewed preprints |
| nature.com | 0.95 | Top-tier journal |
| wikipedia.org | 0.90 | Community-verified |
| ieee.org | 0.90 | Engineering standards body |
| acm.org | 0.90 | CS professional society |
| scholar.google.com | 0.85 | Academic search |
| github.com | 0.70 | Open source projects |
| medium.com | 0.50 | Mixed quality blogs |
| reddit.com | 0.30 | Unverified user posts |
| quora.com | 0.30 | Unverified Q&A |
| (unknown) | 0.50 | Neutral default |

**Snippet Quality Heuristics:**
- Length > 200 chars: +0.3 (substantial content)
- Length > 100 chars: +0.2
- Length > 50 chars: +0.1
- Contains numbers/percentages/years: +0.2 (likely factual)
- Contains technical terms (algorithm, model, data, etc.): +0.05 per term, max +0.3
- Length < 20 chars: -0.2 (penalize trivially short)

**Why No LLM Call:**
The quality gate is pure Python to save budget. Domain trust and snippet quality are
strong enough signals for routing decisions. This saves ~500-1000 tokens per pipeline run.

---

#### `src/agents/analyst.py` - Claim Extraction with Evidence

**Purpose:** Extracts structured claims from sources with evidence linking and
confidence ratings using `with_structured_output(AnalystOutput)`.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `analyst(state)` | Agent fn | Returns {key_claims, conflicts, token_count, pipeline_trace} |

**LLM Call Details:**
- Structured output: `llm.with_structured_output(AnalystOutput)`
- Input: Top 10 sources, each truncated to 300 chars
- Output: 5-8 claims, each with {claim, source_idx, confidence, evidence}
- Also detects cross-source contradictions
- Token estimate: ~1200 tokens per call

**Claim Schema (per claim):**
```python
{
    "claim": "AI agents use LLM-based reasoning",     # The factual assertion
    "source_idx": 1,                                    # 1-based source reference
    "confidence": "high",                               # high, medium, or low
    "evidence": "Source 1 states that agents..."        # Supporting quote/paraphrase
}
```

---

#### `src/agents/synthesizer.py` - Cross-Source Synthesis

**Purpose:** Cross-references claims across sources, detects conflicts between sources,
ranks source reliability, and produces a unified narrative for the writer.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `synthesizer(state)` | Agent fn | Returns {synthesis, token_count, pipeline_trace} |

**LLM Call Details:**
- Standard invoke (NOT structured output): returns free-form narrative text
- Input: Claims with confidence + conflicts + source list
- Output: 3-5 paragraphs of synthesis grouping related claims, noting agreements
  and disagreements, highlighting strong findings, flagging weak evidence
- Token estimate: ~800 tokens (extracted from response metadata)

---

#### `src/agents/writer.py` - Versioned Report Generation

**Purpose:** Writes structured reports with version tracking. Supports both initial
draft generation and revision based on reviewer feedback.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `writer(state)` | Agent fn | Returns {drafts, current_draft, revision_count, token_count, pipeline_trace} |
| `_build_initial_prompt(state)` | Helper | Builds prompt for first draft from claims + synthesis |
| `_build_revision_prompt(state)` | Helper | Builds prompt for revision from current draft + reviewer feedback |

**Versioning:**
- First call (revision_count=0): Uses claims + synthesis to write fresh report
- Subsequent calls (revision_count>0): Uses previous draft + reviewer issues/suggestions
- Each draft tracked as: `{version, content, char_count, pii_scrubbed}`
- All drafts stored in `state.drafts` list for audit trail

**PII Scrubbing:**
The writer is the only agent that applies `scrub_pii()` to its output. Any emails,
phone numbers, or SSNs in the LLM response are replaced with `[REDACTED_TYPE]` tokens.

**Report Sections (prompted):**
```
## Introduction
## Key Findings
## Analysis
## Conclusion
## Sources
```

**Token estimate:** ~1500 tokens per call

---

#### `src/agents/reviewer.py` - Draft Quality Assessment

**Purpose:** Scores report quality 1-10, flags specific issues, suggests improvements.
Controls the refinement loop: passed=True routes to END, False routes back to writer.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `reviewer(state)` | Agent fn | Returns {review, final_report (if passed), token_count, pipeline_trace} |

**LLM Call Details:**
- Structured output: `llm.with_structured_output(ReviewOutput)`
- Input: Current draft (truncated to 3000 chars) + available claims
- Output: {score (1-10), issues (list), suggestions (list), passed (bool)}
- Token estimate: ~800 tokens per call

**Scoring Criteria (prompted):**
- Accuracy: Are claims supported by cited sources? (1-3 points)
- Completeness: Are all major aspects covered? (1-3 points)
- Structure: Is the report well-organized? (1-2 points)
- Citations: Are sources properly referenced? (1-2 points)

**Pass/Fail Logic:**
```python
# Force-pass if max revisions reached (prevent infinite loops)
force_pass = revision_count >= max_revisions
passed = result.passed or force_pass

# If passed, promote draft to final report
if passed:
    output["final_report"] = current_draft
```

**Error Handling:**
On LLM error, the reviewer accepts the current draft as-is (passed=True) to prevent
the pipeline from hanging. This is a deliberate "fail-open" choice, since a draft
exists and is better than nothing.

---

#### `src/tools/search.py` - Tavily Web Search

**Purpose:** Wraps the Tavily Search API for web search. Lazy-initializes the client
so the TAVILY_API_KEY is only needed at runtime, not import time.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `web_search(query, max_results=5)` | Function | Returns list of {title, url, snippet, date, tool} dicts |
| `_get_search()` | Helper | Lazy-initializes TavilySearch singleton |

**Source Dict Schema:**
```python
{
    "title": "AI Agents Survey",
    "url": "https://arxiv.org/abs/2401.0001",
    "snippet": "A comprehensive study of AI agent architectures...",  # max 500 chars
    "date": "2025-01-15",
    "tool": "tavily"
}
```

---

#### `src/tools/wikipedia.py` - Wikipedia REST API Search

**Purpose:** Searches Wikipedia using the public REST API. No API key required.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `wiki_search(query, max_results=3)` | Function | Returns list of {title, url, snippet, date, tool} dicts |

**API Endpoints:**
- Search: `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}`
- Returns titles, snippets (HTML cleaned), and timestamps
- User-Agent: "ResearchBot/1.0"

---

#### `src/tools/scraper.py` - Web Scraper

**Purpose:** Extracts text content from web pages using BeautifulSoup. Validates URLs
before fetching and strips navigation/script elements.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `scrape_url(url, max_chars=3000)` | Function | Returns {title, url, snippet, date, tool} or empty dict |
| `scrape_urls(urls, max_chars=3000)` | Function | Batch scraping, skips failures |

**Extraction Process:**
1. Validate URL via HEAD request (5s timeout)
2. GET full page content
3. Remove `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags
4. Extract title from `<title>` tag
5. Get text content, truncated to `max_chars`

---

#### `src/tools/tool_selector.py` - Query-Type to Tool Routing

**Purpose:** Decides which research tools to use based on query keyword analysis.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `select_tools(query)` | Function | Returns list of tool names: ["tavily"], ["wikipedia"], or both |

**Routing Logic:**

| Signal | Keywords | Tool Selection |
|--------|----------|----------------|
| Web only | "latest", "recent", "2024", "2025", "2026", "new", "current", "trending" | ["tavily"] |
| Wiki signal | "what is", "define", "history of", "overview", "explain", "who is" | ["tavily", "wikipedia"] |
| Both signals | Mixed keywords | ["tavily", "wikipedia"] |
| No signal | No matching keywords | ["tavily"] (default) |

**Important:** Tavily is ALWAYS included as the primary tool. Wikipedia is added as a
supplement for factual/definitional queries.

---

#### `src/cache/research_cache.py` - SQLite Source & Query Cache

**Purpose:** Caches research results in SQLite to avoid duplicate API calls. Uses
exact query matching (SHA-256 hash), NOT vector similarity.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `get_cached_sources(query)` | Function | Returns cached sources or None (respects 24h TTL) |
| `cache_sources(query, sources)` | Function | Stores query results + updates source index |
| `get_cache_stats()` | Function | Returns {cached_queries, indexed_sources, total_hits} |
| `clear_cache()` | Function | Deletes all cache entries |
| `_hash_query(query)` | Helper | SHA-256 hash, case-insensitive, first 16 chars |

**Cache Behavior:**
- Query hash: `sha256(query.strip().lower())[:16]`
- TTL: 24 hours (CACHE_TTL = 3600 * 24 seconds)
- On hit: Increments hit_count, returns cached sources
- On miss: Returns None
- On expired: Deletes stale entry, returns None
- DB path: `data/research_cache.db` (auto-created)

---

#### `src/models/state.py` - Shared State Schema

**Purpose:** Defines the ResearchState TypedDict that flows through the entire pipeline,
plus Pydantic schemas for structured output validation.

See [Section 4](#4-shared-state-schema) for full details.

---

#### `src/config.py` - YAML Configuration Loader

**Purpose:** Loads and caches configuration from `configs/base.yaml`. Provides typed
accessors for model, pipeline, agent, and budget settings.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `load_config(path=None)` | Function | Loads and caches YAML config (singleton pattern) |
| `get_model_name()` | Function | Returns configured LLM model name (default: "gemini-2.5-flash") |
| `get_agent_config(agent_name)` | Function | Returns per-agent config (temperature, max_tokens) |
| `get_pipeline_config()` | Function | Returns pipeline thresholds (max_sub_topics, quality_threshold, etc.) |
| `get_budget_config()` | Function | Returns budget settings (token_budget, warn_at_percent) |

**Fallback Config:**
If the YAML file is missing, `_default_config()` provides safe defaults:
```python
{
    "model": {"name": "gemini-2.5-flash", "provider": "google"},
    "budget": {"token_budget": 50000, "warn_at_percent": 80},
    "search": {"max_results": 5, "timeout_seconds": 10},
    "pipeline": {
        "max_sub_topics": 3,
        "max_revisions": 2,
        "quality_threshold": 0.4,
        "review_pass_score": 7,
    },
}
```

---

#### `src/guardrails.py` - Safety Layer

**Purpose:** Production safety checks including PII detection/scrubbing, URL validation,
token budget enforcement, and API rate limiting.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `detect_pii(text)` | Function | Scans text for PII, returns {pii_type: [matches]} |
| `scrub_pii(text)` | Function | Replaces PII with [REDACTED_TYPE], returns (cleaned_text, types_found) |
| `validate_url(url, timeout=5)` | Function | HEAD request to check URL reachability |
| `check_budget(current, budget=50000)` | Function | Returns True if under budget |
| `detect_injection(text)` | Function | Scans text for prompt injection patterns, returns (is_safe, matches) |
| `RateLimiter(max_rpm=30)` | Class | Token bucket rate limiter for API calls |
| `rate_limiter` | Global | Singleton RateLimiter instance (30 RPM) |
| `check_all_guardrails(state)` | Function | Runs all checks (budget, PII, injection, rate), returns summary dict |

**PII Patterns (compiled regex):**

| Type | Pattern | Example Match |
|------|---------|---------------|
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | john@example.com |
| Phone | `\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b` | 555-123-4567 |
| SSN | `\b\d{3}-\d{2}-\d{4}\b` | 123-45-6789 |

**Rate Limiter:**
```python
class RateLimiter:
    def __init__(self, max_rpm=30):
        self.min_interval = 60.0 / max_rpm  # 2 seconds between calls
        self._last_call = 0.0
        self._lock = threading.Lock()       # Thread-safe

    def wait_if_needed(self) -> float:
        # Blocks until safe to call, returns seconds waited
```

---

#### `app.py` - Streamlit UI

**Purpose:** Interactive web interface for the research pipeline. Features live agent-by-agent
execution with streaming status updates, graph topology display, pipeline trace timeline,
source quality heatmap, and an evaluation tab.

**Key Components:**

| Component | Type | Description |
|-----------|------|-------------|
| `render_sidebar()` | Function | Model info, API key status, token budget bar, guardrail status, cache stats |
| `render_research_tab()` | Function | Query input, pipeline execution, results display |
| `run_research_pipeline(query)` | Function | Streams pipeline via `app.stream()`, renders each agent's output |
| `render_node_output(name, state)` | Function | Agent-specific rendering (claims, rankings, drafts, scores) |
| `render_pipeline_trace(trace)` | Function | Timeline table with duration, tokens, summary per agent |
| `display_final_report(result)` | Function | Markdown report + summary metrics |
| `render_evaluation_tab()` | Function | Single vs multi-agent comparison UI |

**Session State Keys:**
- `research_result` - Full pipeline result dict
- `eval_results` - Evaluation comparison results
- `token_count` - Running token total
- `guardrail_status` - {pii, url, budget} status indicators

---

## 3. LLM and Tool Integration

### LLM: Google Gemini 2.5 Flash

All agent LLM calls use the same model via LangChain's `ChatGoogleGenerativeAI`.

| Setting | Value |
|---------|-------|
| Model | `gemini-2.5-flash` |
| Provider | Google AI (via `langchain_google_genai`) |
| Temperature | 0 (deterministic, all agents) |
| API Key | `GOOGLE_API_KEY` environment variable |
| Rate Limit | 30 requests per minute (free tier) |
| Daily Limit | 1500 requests per day (free tier) |
| Cost | $0.00 (free tier) |

**Which Agents Call the LLM:**

| Agent | LLM Call? | Method | Pydantic Schema |
|-------|-----------|--------|-----------------|
| Planner | Yes | `with_structured_output(PlannerOutput)` | PlannerOutput |
| Researcher | **No** | Calls Tavily/Wikipedia APIs only | - |
| Quality Gate | **No** | Pure Python heuristics | - |
| Retry Researcher | **No** | Reuses researcher function | - |
| Analyst | Yes | `with_structured_output(AnalystOutput)` | AnalystOutput |
| Synthesizer | Yes | `llm.invoke()` (free-form text) | - |
| Writer | Yes | `llm.invoke()` (free-form text) | - |
| Reviewer | Yes | `with_structured_output(ReviewOutput)` | ReviewOutput |

**Total LLM calls per pipeline run:**
- Happy path: 5 calls (planner + analyst + synthesizer + writer + reviewer)
- With quality retry: 5 calls (retry researcher does not call LLM)
- With 1 revision: 7 calls (+ writer + reviewer)
- With 2 revisions: 9 calls (+ 2x writer + 2x reviewer)
- Worst case: 9 calls at 6-second spacing = ~54 seconds of rate-limited execution

### Tool: Tavily Search API

| Setting | Value |
|---------|-------|
| Provider | Tavily (via `langchain_tavily.TavilySearch`) |
| API Key | `TAVILY_API_KEY` environment variable |
| Max results | 5 per query |
| Snippet length | 500 chars max |
| Free tier | 1,000 searches/month |
| Initialization | Lazy (only when first search is needed) |

### Tool: Wikipedia REST API

| Setting | Value |
|---------|-------|
| Endpoint | `https://en.wikipedia.org/w/api.php` |
| API Key | Not required (public API) |
| Max results | 3 per query |
| Snippet length | 500 chars max |
| Timeout | 10 seconds |
| User-Agent | "ResearchBot/1.0" |

### Tool: BeautifulSoup Scraper

| Setting | Value |
|---------|-------|
| Library | BeautifulSoup4 (html.parser) |
| Max content | 3,000 chars |
| URL validation | HEAD request, 5s timeout |
| Stripped elements | script, style, nav, footer, header |

### Tool Selection Routing

The `tool_selector.py` module uses keyword-based heuristics to decide which tools
to use for each query:

```
Query: "What are the latest trends in AI agents?"
  Signals: "what are" (wiki) + "latest" (web)
  Both signals -> tools = ["tavily", "wikipedia"]

Query: "Latest breakthroughs in protein folding"
  Signals: "latest" (web), "breakthroughs" (no match)
  Web only -> tools = ["tavily"]

Query: "What is retrieval augmented generation?"
  Signals: "what is" (wiki)
  Wiki signal, no web signal -> tools = ["tavily", "wikipedia"]

Query: "Compare LangGraph and CrewAI"
  Signals: none
  No signal -> tools = ["tavily"]  (default)
```

### SQLite Research Cache

| Setting | Value |
|---------|-------|
| Database | `data/research_cache.db` (auto-created) |
| Library | Python `sqlite3` (stdlib) |
| TTL | 24 hours |
| Hash | SHA-256, first 16 chars, case-insensitive |
| Tables | `query_cache`, `source_index` |

---

## 4. Shared State Schema

The `ResearchState` TypedDict is the single shared state flowing through the entire
LangGraph pipeline. Each agent reads from and writes to specific fields.

```python
class ResearchState(TypedDict):
    # --- Input ---
    query: str                                           # User's research question

    # --- Planner Output ---
    sub_topics: list[str]                                # 1-3 focused sub-topics
    research_plan: str                                   # Brief strategy description

    # --- Researcher Output (parallel, merged via operator.add) ---
    sources: Annotated[list[dict], operator.add]         # All found sources
    search_queries_used: Annotated[list[str], operator.add]  # Queries sent to tools

    # --- Quality Gate Output ---
    quality_score: float                                 # 0.0 - 1.0 aggregate score
    quality_passed: bool                                 # True if score >= threshold

    # --- Analyst Output ---
    key_claims: list[dict]                               # 5-8 claims with evidence
    conflicts: list[dict]                                # Cross-source contradictions

    # --- Synthesizer Output ---
    synthesis: str                                       # Unified narrative
    source_ranking: list[dict]                           # Sources ranked by quality

    # --- Writer Output (versioned) ---
    drafts: list[dict]                                   # All draft versions
    current_draft: str                                   # Latest draft text

    # --- Reviewer Output ---
    review: dict                                         # {score, issues, suggestions, passed}
    revision_count: int                                  # Current revision number

    # --- Pipeline Metadata ---
    token_count: Annotated[int, operator.add]            # Cumulative tokens (parallel-safe)
    errors: Annotated[list[str], operator.add]           # Error log (append-only)
    final_report: str                                    # Promoted from current_draft on pass
    pipeline_trace: Annotated[list[dict], operator.add]  # Execution trace per agent
```

### Fields Using `operator.add` Reducer

Four fields use `Annotated[..., operator.add]` to support parallel execution via Send().
When multiple researcher instances run in parallel, their return values for these fields
are automatically merged (concatenated) by LangGraph:

| Field | Type | Why operator.add |
|-------|------|------------------|
| `sources` | `list[dict]` | Multiple researchers each return their own sources |
| `search_queries_used` | `list[str]` | Track which queries each researcher used |
| `token_count` | `int` | Sum token usage across parallel agents |
| `errors` | `list[str]` | Collect errors from all parallel agents |
| `pipeline_trace` | `list[dict]` | Merge trace entries from parallel agents |

### Pydantic Schemas for Structured Output

```python
class PlannerOutput(BaseModel):
    sub_topics: list[str]    # 1-3 items (min_length=1, max_length=3)
    research_plan: str       # Brief strategy (2-3 sentences)

class ClaimOutput(BaseModel):
    claim: str               # The factual assertion
    source_idx: int          # 1-based source reference
    confidence: str          # "high", "medium", or "low"
    evidence: str            # Supporting quote or paraphrase

class AnalystOutput(BaseModel):
    claims: list[ClaimOutput]  # 5-8 extracted claims
    conflicts: list[str]       # Cross-source contradictions

class ReviewOutput(BaseModel):
    score: int               # 1-10 (ge=1, le=10)
    issues: list[str]        # Specific problems to fix
    suggestions: list[str]   # Concrete improvements
    passed: bool             # True if publication-ready
```

### Default State Initialization

```python
def default_state(query: str) -> dict:
    return {
        "query": query,
        "sub_topics": [],
        "research_plan": "",
        "sources": [],
        "search_queries_used": [],
        "quality_score": 0.0,
        "quality_passed": False,
        "key_claims": [],
        "conflicts": [],
        "synthesis": "",
        "source_ranking": [],
        "drafts": [],
        "current_draft": "",
        "review": {},
        "revision_count": 0,
        "token_count": 0,
        "errors": [],
        "final_report": "",
        "pipeline_trace": [],
    }
```

### Which Agent Reads/Writes Which Fields

```
                 READS                          WRITES
Planner:         query                    -->   sub_topics, research_plan
Researcher:      query, token_count       -->   sources, search_queries_used
Quality Gate:    sources                  -->   quality_score, quality_passed, source_ranking
Analyst:         sources, token_count     -->   key_claims, conflicts
Synthesizer:     key_claims, conflicts,   -->   synthesis
                 sources
Writer:          key_claims, synthesis,   -->   drafts, current_draft, revision_count
                 sources, review, drafts
Reviewer:        current_draft,           -->   review, final_report (if passed)
                 key_claims, revision_count
```

---

## 5. Complete Research Query Flow - Happy Path

This traces a complete successful pipeline execution with example state values at each step.

**Query:** "What are the latest trends in AI agents for 2025?"

---

### Step 1: User Submits Query via Streamlit UI

```python
# app.py - render_research_tab()
query = "What are the latest trends in AI agents for 2025?"
initial_state = default_state(query)
app = build_graph()
```

**State after initialization:**
```python
{
    "query": "What are the latest trends in AI agents for 2025?",
    "sub_topics": [],
    "sources": [],
    "quality_score": 0.0,
    "quality_passed": False,
    "token_count": 0,
    "errors": [],
    "pipeline_trace": [],
    # ... all other fields at defaults
}
```

---

### Step 2: Guardrails Pre-Check

Before the first agent runs, the planner checks budget:
```python
check_budget(state.get("token_count", 0))  # check_budget(0) -> True
```

The rate limiter is checked before each LLM call:
```python
rate_limiter.wait_if_needed()  # First call, no wait needed
```

---

### Step 3: Planner Decomposes Query

The planner calls Gemini with `with_structured_output(PlannerOutput)`:

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
structured_llm = llm.with_structured_output(PlannerOutput)
result = structured_llm.invoke(
    "You are a research planner. Break this research query into "
    "1-3 focused sub-topics that can be researched independently..."
)
```

**Planner returns:**
```python
{
    "sub_topics": [
        "What are the leading AI agent frameworks in 2025?",
        "How do multi-agent systems differ from single-agent architectures?"
    ],
    "research_plan": "Research leading frameworks (LangGraph, CrewAI, AutoGen) "
                     "and compare multi-agent vs single-agent patterns.",
    "token_count": 500,
    "pipeline_trace": [{
        "agent": "planner",
        "duration_ms": 2100,
        "tokens": 500,
        "summary": "Decomposed into 2 sub-topics"
    }]
}
```

**State after planner:**
```python
{
    "query": "What are the latest trends in AI agents for 2025?",
    "sub_topics": [
        "What are the leading AI agent frameworks in 2025?",
        "How do multi-agent systems differ from single-agent architectures?"
    ],
    "research_plan": "Research leading frameworks...",
    "token_count": 500,    # Accumulated
    "pipeline_trace": [{"agent": "planner", ...}]
}
```

---

### Step 4: Send() Creates 2 Parallel Researcher Instances

The `route_to_researchers` function creates Send objects:

```python
def route_to_researchers(state):
    sub_topics = state["sub_topics"]  # 2 topics
    return [
        Send("researcher", {"query": "What are the leading AI agent frameworks in 2025?", "token_count": 500}),
        Send("researcher", {"query": "How do multi-agent systems differ from single-agent architectures?", "token_count": 500}),
    ]
```

Both researchers run **in parallel**.

---

### Step 5: Researchers Run in Parallel

**Researcher #1** (topic: "What are the leading AI agent frameworks in 2025?"):
```python
tools = select_tools("What are the leading AI agent frameworks in 2025?")
# "latest" not found, "what are" triggers wiki -> tools = ["tavily", "wikipedia"]

# Tavily returns 5 results, Wikipedia returns 3
# After URL dedup: 7 unique sources
```

**Researcher #1 returns:**
```python
{
    "sources": [
        {"title": "LangGraph Documentation", "url": "https://github.com/langchain-ai/langgraph",
         "snippet": "LangGraph is a framework for building stateful multi-agent...", "tool": "tavily"},
        {"title": "CrewAI Framework", "url": "https://github.com/joaomdmoura/crewai",
         "snippet": "CrewAI enables role-based agent orchestration...", "tool": "tavily"},
        {"title": "AI Agents Survey 2025", "url": "https://arxiv.org/abs/2501.12345",
         "snippet": "This survey covers 15 agent frameworks including...", "tool": "tavily"},
        {"title": "Intelligent agent", "url": "https://en.wikipedia.org/wiki/Intelligent_agent",
         "snippet": "An intelligent agent observes its environment...", "tool": "wikipedia"},
        # ... 3 more sources
    ],
    "search_queries_used": ["What are the leading AI agent frameworks in 2025?"],
    "token_count": 300,
    "pipeline_trace": [{"agent": "researcher", "duration_ms": 3200, "tokens": 300,
                         "summary": "Found 7 sources for 'What are the leading AI agent...'"}]
}
```

**Researcher #2** (similar structure, 5 sources found for multi-agent topic)

**State after parallel merge (operator.add):**
```python
{
    "sources": [... 12 total sources from both researchers ...],
    "search_queries_used": [
        "What are the leading AI agent frameworks in 2025?",
        "How do multi-agent systems differ from single-agent architectures?"
    ],
    "token_count": 1100,    # 500 (planner) + 300 + 300 (2 researchers)
    "pipeline_trace": [
        {"agent": "planner", ...},
        {"agent": "researcher", ...},   # Researcher #1
        {"agent": "researcher", ...},   # Researcher #2
    ]
}
```

---

### Step 6: Quality Gate Scores Sources

The quality gate scores all 12 sources without an LLM call:

```python
# Example scoring for one source:
# URL: https://arxiv.org/abs/2501.12345
#   domain_score = 0.95 (arxiv.org)
#   snippet_score = 0.50 (200+ chars, has numbers, has tech terms)
#   combined = 0.6 * 0.95 + 0.4 * 0.50 = 0.77

# URL: https://github.com/langchain-ai/langgraph
#   domain_score = 0.70 (github.com)
#   snippet_score = 0.45 (has tech terms)
#   combined = 0.6 * 0.70 + 0.4 * 0.45 = 0.60

# Average of top 5 combined scores:
# quality_score = (0.77 + 0.72 + 0.68 + 0.65 + 0.60) / 5 = 0.684
# threshold = 0.4 -> PASS
```

**Quality Gate returns:**
```python
{
    "quality_score": 0.684,
    "quality_passed": True,
    "source_ranking": [
        {"index": 2, "title": "AI Agents Survey 2025", "url": "https://arxiv.org/...",
         "domain_score": 0.95, "snippet_score": 0.50, "combined_score": 0.77},
        # ... sorted descending by combined_score
    ],
    "pipeline_trace": [{
        "agent": "quality_gate", "duration_ms": 0, "tokens": 0,
        "summary": "Score 0.68/0.4 - PASS"
    }]
}
```

**Routing decision:** `route_after_quality` sees `quality_passed=True`, returns `"analyst"`.

---

### Step 7: Analyst Extracts Claims

```python
# Formats top 10 sources as numbered list:
# [1] LangGraph Documentation: LangGraph is a framework...
# [2] CrewAI Framework: CrewAI enables role-based...
# [3] AI Agents Survey 2025: This survey covers 15...
# ...

result = structured_llm.invoke("Extract 5-8 key claims...")
```

**Analyst returns:**
```python
{
    "key_claims": [
        {"claim": "LangGraph is the leading framework for stateful multi-agent applications",
         "source_idx": 1, "confidence": "high",
         "evidence": "Source 1 describes LangGraph as a framework for building stateful multi-agent apps"},
        {"claim": "Multi-agent systems improve research quality by 33% over single-agent",
         "source_idx": 3, "confidence": "high",
         "evidence": "The 2025 survey found 33% completeness improvement"},
        {"claim": "CrewAI uses role-based agent orchestration",
         "source_idx": 2, "confidence": "high",
         "evidence": "Source 2 states CrewAI enables role-based agent orchestration"},
        {"claim": "Tool use via MCP is emerging as a standard",
         "source_idx": 5, "confidence": "medium",
         "evidence": "Source 5 mentions MCP as a growing protocol"},
        {"claim": "Cost optimization requires per-agent token tracking",
         "source_idx": 7, "confidence": "medium",
         "evidence": "Source 7 recommends tracking token usage per agent"},
    ],
    "conflicts": [
        {"description": "Source 1 and Source 6 disagree on whether CrewAI or LangGraph has more adoption"}
    ],
    "token_count": 1200,
    "pipeline_trace": [{"agent": "analyst", "duration_ms": 3400, "tokens": 1200,
                         "summary": "Extracted 5 claims, 1 conflicts"}]
}
```

**State token_count: 1100 + 1200 = 2300**

---

### Step 8: Synthesizer Cross-References

The synthesizer receives claims, conflicts, and sources, then produces a unified narrative:

```python
response = llm.invoke(
    "Group related claims into themes, note agreements and disagreements..."
)
```

**Synthesizer returns:**
```python
{
    "synthesis": "The AI agent landscape in 2025 is dominated by three major frameworks: "
                 "LangGraph, CrewAI, and AutoGen. Multiple sources agree that LangGraph leads "
                 "in stateful multi-agent applications, while CrewAI excels at role-based "
                 "orchestration...\n\nA key finding across sources is the 33% improvement "
                 "in research quality when using multi-agent pipelines compared to single-agent "
                 "approaches. However, this comes at 2.8x higher cost...\n\n"
                 "One conflict was identified: sources disagree on relative adoption rates "
                 "between CrewAI and LangGraph...",
    "token_count": 800,
    "pipeline_trace": [{"agent": "synthesizer", "duration_ms": 2800, "tokens": 800,
                         "summary": "Synthesized 5 claims into narrative"}]
}
```

**State token_count: 2300 + 800 = 3100**

---

### Step 9: Writer Produces Draft v1

The writer uses claims + synthesis to produce a structured report:

```python
# _build_initial_prompt builds:
# "You are a technical writer. Write a structured research report.
#  Use ONLY these verified claims and sources..."
```

**Writer returns:**
```python
{
    "drafts": [{
        "version": 1,
        "content": "## Introduction\nAI agents have evolved from simple chatbot wrappers...\n\n"
                   "## Key Findings\n1. **LangGraph leads** in stateful multi-agent... [Source 1]\n"
                   "2. **33% quality improvement** with multi-agent... [Source 3]\n...\n\n"
                   "## Analysis\nThe shift toward multi-agent systems reflects...\n\n"
                   "## Conclusion\n...\n\n## Sources\n[1] LangGraph Documentation...",
        "char_count": 3200,
        "pii_scrubbed": []    # No PII found in this draft
    }],
    "current_draft": "## Introduction\nAI agents have evolved...",
    "revision_count": 1,
    "token_count": 1500,
    "pipeline_trace": [{"agent": "writer", "duration_ms": 3100, "tokens": 1500,
                         "summary": "Draft v1: 3200 chars"}]
}
```

**State token_count: 3100 + 1500 = 4600**

---

### Step 10: Reviewer Scores Draft - PASS

```python
result = structured_llm.invoke(
    "Score this report 1-10. Criteria: accuracy, completeness, structure, citations..."
)
```

**Reviewer returns:**
```python
{
    "review": {
        "score": 8,
        "issues": [],
        "suggestions": ["Consider adding a section on cost trade-offs"],
        "passed": True
    },
    "final_report": "## Introduction\nAI agents have evolved...",  # Promoted!
    "token_count": 800,
    "pipeline_trace": [{"agent": "reviewer", "duration_ms": 2600, "tokens": 800,
                         "summary": "Score 8/10 - PASS"}]
}
```

**Routing decision:** `route_after_review` sees `review.passed=True`, returns `END`.

---

### Step 11: Pipeline Ends - Final State

```python
{
    "query": "What are the latest trends in AI agents for 2025?",
    "sub_topics": ["What are the leading AI agent frameworks...", "How do multi-agent..."],
    "research_plan": "Research leading frameworks...",
    "sources": [... 12 sources ...],
    "search_queries_used": ["What are the leading...", "How do multi-agent..."],
    "quality_score": 0.684,
    "quality_passed": True,
    "key_claims": [... 5 claims ...],
    "conflicts": [{"description": "Source 1 and Source 6 disagree..."}],
    "synthesis": "The AI agent landscape in 2025...",
    "source_ranking": [... 12 ranked sources ...],
    "drafts": [{"version": 1, "content": "...", "char_count": 3200, "pii_scrubbed": []}],
    "current_draft": "## Introduction\nAI agents have evolved...",
    "review": {"score": 8, "issues": [], "suggestions": [...], "passed": True},
    "revision_count": 1,
    "token_count": 5400,     # 500+300+300+0+1200+800+1500+800
    "errors": [],
    "final_report": "## Introduction\nAI agents have evolved...",
    "pipeline_trace": [
        {"agent": "planner",       "duration_ms": 2100, "tokens": 500},
        {"agent": "researcher",    "duration_ms": 3200, "tokens": 300},
        {"agent": "researcher",    "duration_ms": 2900, "tokens": 300},
        {"agent": "quality_gate",  "duration_ms": 0,    "tokens": 0},
        {"agent": "analyst",       "duration_ms": 3400, "tokens": 1200},
        {"agent": "synthesizer",   "duration_ms": 2800, "tokens": 800},
        {"agent": "writer",        "duration_ms": 3100, "tokens": 1500},
        {"agent": "reviewer",      "duration_ms": 2600, "tokens": 800},
    ]
}
```

### Step 12: Final Report Displayed in Streamlit

```python
# app.py - display_final_report(result)
st.markdown(result["final_report"])

# Summary metrics row:
# Sources: 12 | Claims: 5 | Quality: 0.68 | Drafts: 1 | Tokens: 5,400
```

**Happy path total: 5 LLM calls, ~5,400 tokens, ~18 seconds**

---

## 6. Complete Research Query Flow - Quality Gate Retry

This traces what happens when the initial sources are low quality.

**Query:** "What is machine learning?" (but researchers happen to return poor sources)

---

### Step 1-4: Planner + Researcher (same as happy path)

Planner decomposes into 2 sub-topics. Researchers run in parallel.

---

### Step 5: Researchers Return Low-Quality Sources

Due to Tavily returning forum posts instead of academic sources:

```python
{
    "sources": [
        {"title": "ML is cool", "url": "https://reddit.com/r/ml/abc",
         "snippet": "yeah ML is pretty cool lol", "tool": "tavily"},
        {"title": "What is ML?", "url": "https://quora.com/q/123",
         "snippet": "its like AI but different", "tool": "tavily"},
        {"title": "Blog post", "url": "https://random-blog.xyz/ml",
         "snippet": "Machine learning overview", "tool": "tavily"},
    ],
    "search_queries_used": ["What is machine learning?", "ML algorithms overview"],
    "token_count": 600,  # 300 per researcher
}
```

---

### Step 6: Quality Gate Scores Sources - FAIL

```python
# Source 1: reddit.com
#   domain_score = 0.30
#   snippet_score = 0.0 (< 20 chars after "yeah ML is pretty cool lol" = ~27 chars, +0.1 for >50)
#   Actually: length=27 -> +0.1, no numbers -> +0, no tech terms matched -> +0
#   snippet_score = max(0.1 - 0.0, 0) = 0.1
#   combined = 0.6 * 0.30 + 0.4 * 0.1 = 0.22

# Source 2: quora.com
#   domain_score = 0.30
#   snippet_score ~= 0.0 (very short, no data)
#   combined = 0.6 * 0.30 + 0.4 * 0.0 = 0.18

# Source 3: unknown domain
#   domain_score = 0.50
#   snippet_score ~= 0.1 (short, some tech terms)
#   combined = 0.6 * 0.50 + 0.4 * 0.1 = 0.34

# quality_score = mean(0.34, 0.22, 0.18) = 0.247
# threshold = 0.4 -> FAIL
```

**Quality Gate returns:**
```python
{
    "quality_score": 0.247,
    "quality_passed": False,
    "source_ranking": [
        {"title": "Blog post", "combined_score": 0.34},
        {"title": "ML is cool", "combined_score": 0.22},
        {"title": "What is ML?", "combined_score": 0.18},
    ],
    "pipeline_trace": [{"agent": "quality_gate", "summary": "Score 0.25/0.4 - RETRY"}]
}
```

---

### Step 7: Routing Decision - Retry

```python
def route_after_quality(state):
    quality_passed = False
    search_queries_used = ["What is machine learning?", "ML algorithms overview"]  # len=2
    sub_topics = ["What is machine learning?", "ML algorithms overview"]           # len=2

    # len(queries_used) > len(sub_topics)?  2 > 2?  No -> not yet retried
    return "retry_researcher"
```

---

### Step 8: Retry Researcher Runs Broadened Query

```python
def retry_researcher(state):
    query = "What is machine learning?"
    broad_query = "What is machine learning? comprehensive overview analysis"

    # Calls researcher() with broadened query
    result = researcher({"query": broad_query, "token_count": state["token_count"]})
    # Researcher selects tools: "overview" triggers wiki -> ["tavily", "wikipedia"]
```

**Retry Researcher returns:**
```python
{
    "sources": [
        {"title": "Machine learning", "url": "https://en.wikipedia.org/wiki/Machine_learning",
         "snippet": "Machine learning is a subset of artificial intelligence...", "tool": "wikipedia"},
        {"title": "ML Survey 2025", "url": "https://arxiv.org/abs/2501.99999",
         "snippet": "This comprehensive survey covers 200 machine learning algorithms...", "tool": "tavily"},
        # ... 5 more sources
    ],
    "search_queries_used": ["What is machine learning? comprehensive overview analysis"],
    "token_count": 300,
    "pipeline_trace": [{"agent": "retry_researcher", ...}]
}
```

**State after retry (operator.add merges sources):**
```python
{
    "sources": [
        # Original 3 bad sources + 7 new sources from retry = 10 total
    ],
    "search_queries_used": [
        "What is machine learning?",
        "ML algorithms overview",
        "What is machine learning? comprehensive overview analysis"
    ],
    # ... len(queries) = 3 > len(sub_topics) = 2 -> marks as "already retried"
}
```

---

### Step 9: Pipeline Continues to Analyst

After `retry_researcher`, the graph has a deterministic edge directly to `analyst`
(it does NOT go through quality_gate again). This is intentional - the retry added
better sources, and the combined pool should be sufficient.

```
retry_researcher --> analyst --> synthesizer --> writer --> reviewer --> END
```

The analyst now has 10 sources (3 bad + 7 good), with the better sources ranked higher.

---

### Quality Gate Retry - Complete Token Count

```
Planner:            500 tokens
Researcher #1:      300 tokens
Researcher #2:      300 tokens
Quality Gate:         0 tokens  (pure Python)
Retry Researcher:   300 tokens
Analyst:          1,200 tokens
Synthesizer:        800 tokens
Writer:           1,500 tokens
Reviewer:           800 tokens
---------------------------------
Total:            5,700 tokens  (only 300 more than happy path)
```

---

## 7. Complete Research Query Flow - Reviewer Refinement Loop

This traces what happens when the reviewer rejects the first draft.

**Query:** "Compare transformer and mamba architectures"

---

### Steps 1-8: Normal Pipeline Through Writer v1

Everything proceeds normally through the pipeline. The writer produces draft v1.

**Writer v1 returns:**
```python
{
    "drafts": [{
        "version": 1,
        "content": "## Key Findings\nTransformers use attention...\nMamba uses SSM...",
        "char_count": 1800,
        "pii_scrubbed": []
    }],
    "current_draft": "## Key Findings\nTransformers use attention...",
    "revision_count": 1,
    "token_count": 1500,
}
```

---

### Step 9: Reviewer Rejects Draft v1

```python
result = structured_llm.invoke(
    "Score this report 1-10..."
)
```

**Reviewer returns:**
```python
{
    "review": {
        "score": 5,
        "issues": [
            "Missing citations for 2 claims about Mamba performance",
            "No introduction section",
            "Conclusion is only one sentence"
        ],
        "suggestions": [
            "Add an introduction explaining why this comparison matters",
            "Cite source numbers for all performance claims",
            "Expand conclusion to summarize key trade-offs"
        ],
        "passed": False
    },
    "token_count": 800,
    "pipeline_trace": [{"agent": "reviewer", "summary": "Score 5/10 - REVISE (attempt 1/2)"}]
}
```

---

### Step 10: Routing Decision - Loop Back to Writer

```python
def route_after_review(state):
    review = {"passed": False}
    revision_count = 1
    max_revisions = 2  # from config

    # passed=False AND revision_count(1) < max_revisions(2) -> loop back
    return "writer"
```

---

### Step 11: Writer Produces Draft v2

The writer detects `revision_count > 0` and uses `_build_revision_prompt`:

```python
def _build_revision_prompt(state):
    return (
        "You are a technical writer revising a research report.\n\n"
        f"Current draft:\n{current_draft[:3000]}\n\n"
        f"Reviewer issues to fix:\n"
        f"- Missing citations for 2 claims about Mamba performance\n"
        f"- No introduction section\n"
        f"- Conclusion is only one sentence\n\n"
        f"Reviewer suggestions:\n"
        f"- Add an introduction explaining why this comparison matters\n"
        f"- Cite source numbers for all performance claims\n"
        f"- Expand conclusion to summarize key trade-offs\n\n"
        f"Revise the report to address ALL issues and suggestions."
    )
```

**Writer v2 returns:**
```python
{
    "drafts": [
        {"version": 1, "content": "## Key Findings\n...", "char_count": 1800},
        {"version": 2, "content": "## Introduction\nThe choice between transformer...\n\n"
                                   "## Key Findings\n...[Source 3]...[Source 5]...\n\n"
                                   "## Conclusion\nBoth architectures serve different needs. "
                                   "Transformers excel at attention-heavy tasks while Mamba "
                                   "offers linear scaling...",
         "char_count": 3400, "pii_scrubbed": []}
    ],
    "current_draft": "## Introduction\nThe choice between transformer...",
    "revision_count": 2,
    "token_count": 1500,
    "pipeline_trace": [{"agent": "writer", "summary": "Draft v2: 3400 chars"}]
}
```

---

### Step 12: Reviewer Scores Draft v2 - PASS

```python
{
    "review": {
        "score": 8,
        "issues": [],
        "suggestions": ["Minor: could add a comparison table"],
        "passed": True
    },
    "final_report": "## Introduction\nThe choice between transformer...",
    "token_count": 800,
    "pipeline_trace": [{"agent": "reviewer", "summary": "Score 8/10 - PASS"}]
}
```

**Routing:** `review.passed=True` -> END

---

### Refinement Loop - Complete Token Count

```
Planner:            500 tokens
Researcher #1:      300 tokens
Researcher #2:      300 tokens
Quality Gate:         0 tokens
Analyst:          1,200 tokens
Synthesizer:        800 tokens
Writer v1:        1,500 tokens
Reviewer v1:        800 tokens  <- FAIL, score 5/10
Writer v2:        1,500 tokens  <- Revision
Reviewer v2:        800 tokens  <- PASS, score 8/10
---------------------------------
Total:            7,700 tokens  (7 LLM calls)
Total time:       ~25 seconds
```

---

### What If Draft v2 Also Fails?

If the reviewer also rejects v2 (score < 7):

```python
def route_after_review(state):
    review = {"passed": False}
    revision_count = 2
    max_revisions = 2

    # revision_count(2) >= max_revisions(2) -> END (prevent infinite loop)
    return END
```

The pipeline ends with whatever draft is current. The `final_report` is NOT set by
the reviewer in this case (since `passed=False`), but the `current_draft` is still
available in the state.

Additionally, the reviewer itself has a `force_pass` safety net:
```python
force_pass = revision_count >= max_revisions
passed = result.passed or force_pass  # True even if LLM says False
```

This ensures `final_report` is always populated when the pipeline exits.

---

## 8. Sequence Diagrams

### Happy Path Sequence Diagram

```
User          Streamlit       Graph         Planner      Researcher(s)   QualityGate    Analyst     Synthesizer   Writer      Reviewer
  |               |             |              |              |               |            |             |           |            |
  |--query------->|             |              |              |               |            |             |           |            |
  |               |--invoke---->|              |              |               |            |             |           |            |
  |               |             |--state------>|              |               |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |              |--LLM call--->|              |             |             |           |            |
  |               |             |              |   (Gemini)   |              |             |             |           |            |
  |               |             |              |<--Planner----|              |             |             |           |            |
  |               |             |              |   Output     |              |             |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |<--sub_topics-|              |               |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--Send(topic1)-------------->|               |            |             |           |            |
  |               |             |--Send(topic2)-------------->|  (parallel)   |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |              |              |--Tavily API-->|            |             |           |            |
  |               |             |              |              |--Wiki API---->|            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |<--sources (operator.add)----|               |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--sources---------------------------->|      |            |             |           |            |
  |               |             |              |              |        |      |            |             |           |            |
  |               |             |              |              |  domain_trust  |            |             |           |            |
  |               |             |              |              |  snippet_score |            |             |           |            |
  |               |             |              |              |  (pure Python) |            |             |           |            |
  |               |             |              |              |        |      |            |             |           |            |
  |               |             |<--score=0.68, passed=True-----------|      |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--sources+claims---------------------------->|            |             |           |            |
  |               |             |              |              |               |--LLM call->|             |           |            |
  |               |             |              |              |               |  (Gemini)  |             |           |            |
  |               |             |<--key_claims, conflicts---------------------|            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--claims+conflicts+sources------------------------------>|            |            |
  |               |             |              |              |               |            |--LLM call-->|           |            |
  |               |             |              |              |               |            |  (Gemini)   |           |            |
  |               |             |<--synthesis--------------------------------------------|            |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--claims+synthesis+sources---------------------------------------------->|          |            |
  |               |             |              |              |               |            |             |--LLM call->|           |
  |               |             |              |              |               |            |             | + PII scrub|           |
  |               |             |<--draft v1, revision_count=1-----------------------------------------------|          |            |
  |               |             |              |              |               |            |             |           |            |
  |               |             |--draft+claims--------------------------------------------------------------------->|
  |               |             |              |              |               |            |             |           |--LLM call->
  |               |             |              |              |               |            |             |           |  (Gemini)
  |               |             |<--score=8/10, passed=True, final_report--------------------------------------------|
  |               |             |              |              |               |            |             |           |            |
  |               |             |---END------->|              |               |            |             |           |            |
  |               |<--result----|              |              |               |            |             |           |            |
  |<--report------|             |              |              |               |            |             |           |            |
  |               |             |              |              |               |            |             |           |            |
```

### Quality Gate Retry Sequence Diagram

```
 Researcher(s)    QualityGate      RetryResearcher    Analyst
       |               |                 |               |
       |--sources----->|                 |               |
       |  (reddit,     |                 |               |
       |   quora)      |                 |               |
       |               |                 |               |
       |               |--score=0.25---->|               |
       |               |  FAIL           |               |
       |               |  (< 0.4)        |               |
       |               |                 |               |
       |               |  route_after_   |               |
       |               |  quality:       |               |
       |               |  "retry_        |               |
       |               |   researcher"   |               |
       |               |                 |               |
       |               |--retry--------->|               |
       |               |                 |               |
       |               |                 |--broadened    |
       |               |                 |  query:       |
       |               |                 |  "{query}     |
       |               |                 |  comprehensive|
       |               |                 |  overview     |
       |               |                 |  analysis"    |
       |               |                 |               |
       |               |                 |--Tavily API   |
       |               |                 |--Wiki API     |
       |               |                 |               |
       |               |                 |--new sources  |
       |               |                 |  (arxiv,      |
       |               |                 |   wikipedia)  |
       |               |                 |               |
       |               |                 |--directly---->|
       |               |                 |  to analyst   |
       |               |                 |  (skip QG     |
       |               |                 |   recheck)    |
       |               |                 |               |
       |               |                 |               |--LLM call
       |               |                 |               |  (analyze
       |               |                 |               |   combined
       |               |                 |               |   sources)
       |               |                 |               |
       |               |                 |               |--continues
       |               |                 |               |  to synth
```

### Reviewer Refinement Loop Sequence Diagram

```
 Writer              Reviewer             Writer (rev)        Reviewer (rev)
   |                    |                     |                    |
   |--draft v1--------->|                     |                    |
   |  (1800 chars)      |                     |                    |
   |                    |                     |                    |
   |                    |--LLM call           |                    |
   |                    |  score: 5/10        |                    |
   |                    |  issues:            |                    |
   |                    |   "Missing          |                    |
   |                    |    citations"       |                    |
   |                    |   "No intro"        |                    |
   |                    |  passed: False      |                    |
   |                    |                     |                    |
   |                    |  route_after_       |                    |
   |                    |  review:            |                    |
   |                    |  rev=1 < max=2      |                    |
   |                    |  -> "writer"        |                    |
   |                    |                     |                    |
   |                    |--feedback---------->|                    |
   |                    |  (issues +          |                    |
   |                    |   suggestions)      |                    |
   |                    |                     |                    |
   |                    |                     |--revision prompt   |
   |                    |                     |  "Fix these        |
   |                    |                     |   issues..."       |
   |                    |                     |                    |
   |                    |                     |--LLM call          |
   |                    |                     |  (improved draft)  |
   |                    |                     |                    |
   |                    |                     |--draft v2--------->|
   |                    |                     |  (3400 chars)      |
   |                    |                     |                    |
   |                    |                     |                    |--LLM call
   |                    |                     |                    |  score: 8/10
   |                    |                     |                    |  passed: True
   |                    |                     |                    |
   |                    |                     |                    |--final_report
   |                    |                     |                    |  = draft v2
   |                    |                     |                    |
   |                    |                     |                    |--END
```

---

## 9. Database Schema

### SQLite Research Cache

**Database location:** `data/research_cache.db` (auto-created on first access)

```sql
-- Query-level cache: stores complete search results per query
CREATE TABLE IF NOT EXISTS query_cache (
    query_hash  TEXT PRIMARY KEY,    -- SHA-256(query.strip().lower())[:16]
    query       TEXT,                -- Original query text
    sources_json TEXT,               -- JSON-serialized list of source dicts
    created_at  REAL,                -- Unix timestamp (time.time())
    hit_count   INTEGER DEFAULT 0    -- Number of cache hits
);

-- Source-level index: tracks unique URLs across all queries
CREATE TABLE IF NOT EXISTS source_index (
    url         TEXT PRIMARY KEY,    -- Full URL (dedup key)
    title       TEXT,                -- Source title
    snippet     TEXT,                -- First 200 chars of snippet
    first_seen  REAL,                -- Unix timestamp of first discovery
    last_seen   REAL,                -- Unix timestamp of most recent use
    use_count   INTEGER DEFAULT 1    -- Times this URL appeared in results
);
```

### Cache Operations

| Operation | SQL | Description |
|-----------|-----|-------------|
| Cache hit | `SELECT sources_json FROM query_cache WHERE query_hash = ?` | Check if query exists and TTL valid |
| Hit count | `UPDATE query_cache SET hit_count = hit_count + 1` | Increment on each cache hit |
| Cache miss | Returns `None` | Query not found or TTL expired |
| Store results | `INSERT OR REPLACE INTO query_cache ...` | Store/update query results |
| Update source | `INSERT ... ON CONFLICT(url) DO UPDATE SET use_count=use_count+1` | Track URL usage |
| Expire | `DELETE FROM query_cache WHERE query_hash = ?` | Remove when created_at + TTL < now |
| Stats | `SELECT COUNT(*) FROM query_cache` + `SUM(hit_count)` | Dashboard metrics |
| Clear | `DELETE FROM query_cache; DELETE FROM source_index;` | Full reset |

### Cache Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| TTL | 24 hours (86,400 seconds) | `research_cache.py: CACHE_TTL` |
| Hash algorithm | SHA-256, first 16 hex chars | `_hash_query()` |
| Case sensitivity | Case-insensitive (lowered before hash) | `_hash_query()` |
| Snippet storage | First 200 chars | `cache_sources()` |
| DB path | `data/research_cache.db` | `research_cache.py: DB_PATH` |

---

## 10. Performance Metrics

### Pipeline Latency

| Metric | Value | Notes |
|--------|-------|-------|
| Full pipeline (happy path) | 15-20s | 5 LLM calls + 2 search API calls |
| Full pipeline (1 revision) | 20-25s | 7 LLM calls |
| Full pipeline (quality retry + 2 revisions) | 25-35s | 9 LLM calls (worst case) |
| Per-agent LLM call | 2-4s | Gemini 2.5 Flash average |
| Quality Gate (no LLM) | <100ms | Pure Python heuristics |
| Cache hit (SQLite) | <10ms | Local file I/O |
| Tavily search | 1-3s | Network dependent |
| Wikipedia search | 0.5-1.5s | Public API |
| Web scrape | 1-5s | URL validation + page fetch |
| Rate limiter wait | 0-2s | 30 RPM = 2s minimum interval |

### Parallel Execution Speedup

| Configuration | Sequential Time | Parallel Time | Speedup |
|---------------|----------------|---------------|---------|
| 1 sub-topic | 3s | 3s | 1.0x |
| 2 sub-topics | 6s | 3s | 2.0x |
| 3 sub-topics | 9s | 3s | 3.0x |

The `Send()` fan-out enables 2-3 researchers to run concurrently. Each researcher
makes independent API calls (Tavily + Wikipedia), and their results merge via
`operator.add` when all complete.

### Token Usage

| Agent | Tokens/Call | Calls (happy) | Calls (worst) |
|-------|-------------|---------------|----------------|
| Planner | ~500 | 1 | 1 |
| Researcher | ~300 | 2-3 | 2-3 |
| Quality Gate | 0 | 1 | 1 |
| Retry Researcher | ~300 | 0 | 1 |
| Analyst | ~1,200 | 1 | 1 |
| Synthesizer | ~800 | 1 | 1 |
| Writer | ~1,500 | 1 | 3 (1 + 2 revisions) |
| Reviewer | ~800 | 1 | 3 (1 + 2 re-reviews) |
| **Total** | - | **5,400** | **9,200** |

### Cost Per Query

| Configuration | Tokens | Cost | Notes |
|---------------|--------|------|-------|
| Happy path | ~5,400 | $0.00 | Gemini free tier |
| 1 revision | ~7,700 | $0.00 | Gemini free tier |
| 2 revisions | ~9,200 | $0.00 | Gemini free tier |
| Single-agent baseline | ~1,850 | $0.00 | 1 LLM call |

**Note:** All costs are $0.00 because Gemini 2.5 Flash provides a free tier.
At paid pricing, the cost would be approximately $0.001 per query.

### Test Suite Performance

| Metric | Value |
|--------|-------|
| Total tests | 110 |
| All passing | Yes |
| API keys required | No (all mocked) |
| Execution time | ~3 seconds |

---

## 11. Security and Privacy

### PII Detection and Scrubbing

The guardrails module detects and scrubs three types of PII:

| PII Type | Pattern | Replacement |
|----------|---------|-------------|
| Email | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` | `[REDACTED_EMAIL]` |
| Phone | `\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b` | `[REDACTED_PHONE]` |
| SSN | `\b\d{3}-\d{2}-\d{4}\b` | `[REDACTED_SSN]` |

**Where PII scrubbing is applied:**
- Writer agent: `scrub_pii()` is called on every draft before storing in state
- The writer is the only agent that generates free-form text that could contain PII
  from source snippets

**Example:**
```
Input:  "Contact john@example.com or call 555-123-4567 for the study"
Output: "Contact [REDACTED_EMAIL] or call [REDACTED_PHONE] for the study"
Types:  ["email", "phone"]
```

### Token Budget Enforcement

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Total budget | 50,000 tokens | Hard cap across entire pipeline run |
| Check point | Before every LLM call | Each agent checks `check_budget()` first |
| Graceful degradation | Yes | Agents skip LLM call, pass through existing state |
| Budget tracking | `Annotated[int, operator.add]` | Parallel-safe accumulation |
| Warning threshold | 80% (configurable) | Log warning when 80% consumed |

**Budget enforcement flow:**
```python
# Every agent starts with:
if not check_budget(state.get("token_count", 0)):
    return {
        "errors": ["Budget exceeded before {agent_name}"],
        "pipeline_trace": [{"agent": agent_name, "status": "skipped", "reason": "budget"}],
    }
```

### Rate Limiting

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max RPM | 10 | Gemini free tier limit |
| Min interval | 6.0 seconds | `60 / 10 = 6s` between calls |
| Thread safety | `threading.Lock()` | Safe for parallel execution |
| Global instance | `rate_limiter = RateLimiter(max_rpm=10)` | Shared across all agents |

### URL Validation

The scraper validates URLs before fetching:
```python
def validate_url(url, timeout=5):
    resp = requests.head(url, timeout=timeout, allow_redirects=True)
    return resp.status_code < 400
```

### No User Data Persistence

- The SQLite cache stores query hashes and source metadata only
- No user identifiers are stored
- No session data persists between runs
- Cache can be cleared at any time with `clear_cache()`

### Allowed Domains

URL validation optionally restricts to trusted domains:
```yaml
guardrails:
  url_validation:
    allowed_domains:
      - "wikipedia.org"
      - "arxiv.org"
      - "github.com"
      - "tavily.com"
```

---

## 12. Configuration Guide

### Full `configs/base.yaml` Reference

```yaml
# ============================================
# Multi-Agent Research System Configuration
# ============================================
```

#### Model Configuration

```yaml
model:
  name: "gemini-2.5-flash"     # LLM model for all agent calls
  provider: "google"            # LLM provider (google, openai, etc.)
  fallback: "gemini-2.5-flash"  # Fallback model (same for cost control)
```

| Key | Effect | Change Impact |
|-----|--------|---------------|
| `model.name` | Which LLM all agents use | Changes quality, speed, and cost of all LLM calls |
| `model.provider` | Which API to call | Must match the `langchain_*` import in agent files |
| `model.fallback` | Backup if primary fails | Currently unused (same as primary) |

#### Agent Registry

```yaml
agents:
  planner:
    role: "Research Planner"
    temperature: 0
    max_output_tokens: 500

  researcher:
    role: "Research Specialist"
    temperature: 0
    max_output_tokens: 1500

  quality_gate:
    role: "Quality Assessor"
    # No temperature/tokens - pure Python, no LLM call

  analyst:
    role: "Research Analyst"
    temperature: 0
    max_output_tokens: 1500

  synthesizer:
    role: "Research Synthesizer"
    temperature: 0
    max_output_tokens: 1000

  writer:
    role: "Technical Writer"
    temperature: 0
    max_output_tokens: 2000

  reviewer:
    role: "Report Reviewer"
    temperature: 0
    max_output_tokens: 1000
```

| Key | Effect | Change Impact |
|-----|--------|---------------|
| `agents.{name}.temperature` | LLM creativity vs determinism | 0 = deterministic, higher = more varied |
| `agents.{name}.max_output_tokens` | Max response length | Higher = longer but costlier outputs |

#### Pipeline Parameters

```yaml
pipeline:
  max_sub_topics: 3        # Max sub-topics from planner
  max_revisions: 2         # Max writer-reviewer loop iterations
  quality_threshold: 0.4   # Quality gate pass/fail threshold
  review_pass_score: 7     # Reviewer score needed to pass
```

| Key | Effect | Change Impact |
|-----|--------|---------------|
| `max_sub_topics` | Number of parallel researchers | Higher = more sources but more API calls (1-3 range) |
| `max_revisions` | Writer-reviewer loop bound | Higher = better reports but more time/tokens (0-5 range) |
| `quality_threshold` | Quality gate pass/fail | Lower = more lenient (0.0-1.0, default 0.4) |
| `review_pass_score` | Reviewer pass/fail | Lower = more lenient (1-10, default 7) |

**Example tuning scenarios:**

```yaml
# Fast mode (minimize API calls):
pipeline:
  max_sub_topics: 1        # Single researcher
  max_revisions: 0         # No revision loop
  quality_threshold: 0.2   # Very lenient quality gate
  review_pass_score: 3     # Almost always passes

# Quality mode (maximize report quality):
pipeline:
  max_sub_topics: 3        # Maximum parallel research
  max_revisions: 2         # Up to 2 revisions
  quality_threshold: 0.6   # Strict quality gate
  review_pass_score: 8     # High bar for passing
```

#### Token Budget

```yaml
budget:
  token_budget: 50000          # Hard cap across entire pipeline run
  warn_at_percent: 80          # Log warning at 80% consumption
  graceful_degradation: true   # Produce best output when budget hit
```

| Key | Effect | Change Impact |
|-----|--------|---------------|
| `token_budget` | Total token ceiling | Higher = more expensive but allows more complex research |
| `warn_at_percent` | Warning threshold | Triggers logging when X% of budget consumed |
| `graceful_degradation` | Behavior at budget limit | If true, agents skip LLM calls and pass through state |

#### Search Parameters

```yaml
search:
  provider: "tavily"
  max_results: 5               # Results per search query
  snippet_max_chars: 500       # Truncate snippets
  timeout_seconds: 10          # Per-request timeout
```

| Key | Effect | Change Impact |
|-----|--------|---------------|
| `max_results` | Sources per Tavily search | Higher = more sources but slower and more to analyze |
| `snippet_max_chars` | Source snippet length | Higher = more context for analyst but more tokens |
| `timeout_seconds` | API call timeout | Higher = more resilient but slower failure detection |

#### Guardrails

```yaml
guardrails:
  pii_detection:
    enabled: true
    patterns: ["email", "phone", "ssn"]
    action: "redact"           # Options: redact, warn, block

  url_validation:
    enabled: true
    timeout_seconds: 5
    allowed_domains:
      - "wikipedia.org"
      - "arxiv.org"
      - "github.com"
      - "tavily.com"

  content_filtering:
    max_report_chars: 10000    # Truncate final report if too long
```

#### Evaluation

```yaml
evaluation:
  judge_model: "gemini-2.5-flash"
  test_set_size: 10
  default_run_size: 5
  scoring_dimensions: ["accuracy", "completeness", "citations"]
  scale: "0-3"                # 0=missing, 1=poor, 2=adequate, 3=excellent
```

---

## 13. Evaluation Framework

### Architecture

The evaluation framework lives in `evaluation/` and is independent of the main pipeline.
It compares single-agent (1 LLM call) vs multi-agent (3 LLM calls) approaches using
LLM-as-judge scoring.

**Important distinction:** The evaluation's "multi-agent" is a simplified 3-agent chain
(researcher -> analyst -> writer) using direct Gemini API calls, NOT the full 7-agent
LangGraph pipeline. This creates a fair comparison that isolates the value of agent
specialization from framework overhead.

### Test Set

10 research questions across 2 difficulty levels:

| # | Query | Type | Difficulty |
|---|-------|------|------------|
| 1 | What is retrieval-augmented generation? | factual | easy |
| 2 | Compare transformer and mamba architectures | synthesis | hard |
| 3 | Latest breakthroughs in protein folding AI | factual | medium |
| 4 | How do AI agents differ from chatbots? | factual | easy |
| 5 | Trade-offs between fine-tuning and RAG | synthesis | hard |
| 6 | What is constitutional AI? | factual | medium |
| 7 | How does RLHF work in language models? | factual | medium |
| 8 | Compare LangGraph, CrewAI, and AutoGen frameworks | synthesis | hard |
| 9 | What are mixture-of-experts models? | factual | medium |
| 10 | Evaluate the impact of scaling laws on LLM development | synthesis | hard |

### Scoring Dimensions (LLM-as-Judge)

Each report is scored on a 0-3 scale for three dimensions:

| Score | Label | Meaning |
|-------|-------|---------|
| 0 | Missing | No factual claims / topic not addressed / no sources |
| 1 | Poor | Multiple errors / covers only one aspect / no URLs |
| 2 | Adequate | Mostly correct / covers main points / some real sources |
| 3 | Excellent | All claims verifiable / comprehensive / all citations real |

### Results Summary

| Metric | Single-Agent | Multi-Agent | Improvement |
|--------|-------------|-------------|-------------|
| Accuracy (0-3) | 2.2 | 2.6 | +18% |
| Completeness (0-3) | 1.8 | 2.4 | +33% |
| Citations (0-3) | 1.4 | 2.0 | +43% |
| Avg Tokens | 1,850 | 5,200 | 2.8x |
| Avg Latency | 2.1s | 6.8s | 3.2x |
| Avg Cost | $0.0003 | $0.0008 | 2.8x |

**Verdict:** Multi-agent justified for synthesis tasks (+33% completeness improvement).
Route simple factual lookups to single-agent to save 2.8x cost.

---

## 14. Testing Architecture

### Test Suite Overview

| Test File | Tests | Category | API Keys? | Description |
|-----------|-------|----------|-----------|-------------|
| `test_e2e_pipeline.py` | 37 | E2E | No | Full pipeline with mocked LLM calls |
| `test_ui_smoke.py` | 26 | Smoke | No | Streamlit imports and module wiring |
| `test_guardrails.py` | 14 | Unit | No | PII, budget, rate limiter |
| `test_graph.py` | 9 | Unit | No | Graph routing logic |
| `test_quality_gate.py` | 8 | Unit | No | Source scoring heuristics |
| `test_state.py` | 6 | Unit | No | State schema and Pydantic models |
| `test_cache.py` | 5 | Unit | No | SQLite cache operations |
| `test_tools.py` | 5 | Unit | No | Tool selector keyword routing |
| **Total** | **110** | - | **No** | All tests run without API keys |

### E2E Test Classes

| Class | Tests | What It Verifies |
|-------|-------|------------------|
| `TestFullPipelineMocked` | 2 | Complete happy-path flow, correct agent execution order in trace |
| `TestQualityGateRetryFlow` | 5 | Bad sources fail, good sources pass, retry routing, prevent infinite loop |
| `TestReviewerRefinementLoop` | 4 | Score >= 7 ends, score < 7 loops, max revisions cap |
| `TestParallelFanOut` | 4 | Send() per sub-topic, passes token count, fallback to original query |
| `TestBudgetEnforcement` | 4 | Under/over limit, custom budgets |
| `TestPIIScrubbing` | 5 | Email/phone/SSN scrubbing, clean text unchanged, multi-type |
| `TestCacheIntegration` | 2 | Cache roundtrip, stats update |
| `TestStateSchemaE2E` | 3 | All 18+ fields present, correct types, graph compiles |
| `TestScoringHeuristics` | 6 | Domain trust scores, snippet quality assessment |
| `TestRateLimiterE2E` | 2 | Call tracking, reset functionality |

### Mocking Strategy

The E2E tests mock all external dependencies:

```python
@patch("src.agents.researcher.web_search")       # Tavily API
@patch("src.agents.researcher.wiki_search")       # Wikipedia API
@patch("src.agents.researcher.select_tools")      # Tool selection
@patch("src.agents.planner.ChatGoogleGenerativeAI")    # Gemini LLM
@patch("src.agents.analyst.ChatGoogleGenerativeAI")
@patch("src.agents.synthesizer.ChatGoogleGenerativeAI")
@patch("src.agents.writer.ChatGoogleGenerativeAI")
@patch("src.agents.reviewer.ChatGoogleGenerativeAI")
def test_full_pipeline_happy_path(self, ...):
    # All 8 patches provide deterministic responses
    # The graph runs the REAL routing logic with MOCKED data
```

This approach tests the complete orchestration logic (routing, state merging,
conditional edges, loop bounds) without making any real API calls.

---

## 15. Complete Flow Summary

### System Statistics

| Metric | Value |
|--------|-------|
| Total agents | 8 (planner, researcher, quality_gate, retry_researcher, analyst, synthesizer, writer, reviewer) |
| Agents with LLM calls | 5 (planner, analyst, synthesizer, writer, reviewer) |
| Agents without LLM calls | 3 (researcher, quality_gate, retry_researcher) |
| Pydantic-validated outputs | 3 (PlannerOutput, AnalystOutput, ReviewOutput) |
| External APIs | 3 (Gemini, Tavily, Wikipedia) |
| Graph patterns | 3 (Send fan-out, conditional routing, bounded loop) |
| State fields | 18 (5 use operator.add for parallel merge) |
| Safety guardrails | 4 (PII scrub, URL validate, token budget, rate limit) |
| Database tables | 2 (query_cache, source_index) |
| Config parameters | 20+ (YAML-driven, runtime changeable) |
| Test coverage | 110 tests, 0 API keys required |
| Test categories | 4 (unit, graph, e2e, smoke) |

### Pipeline Decision Points

```
Decision Point 1: Tool Selection (per researcher)
  IF query has wiki keywords AND NOT web-only keywords -> ["tavily", "wikipedia"]
  IF query has wiki keywords AND web keywords         -> ["tavily", "wikipedia"]
  ELSE                                                -> ["tavily"]

Decision Point 2: Quality Gate Routing
  IF quality_score >= 0.4    -> proceed to analyst
  IF quality_score < 0.4
    AND not already retried  -> retry_researcher
    AND already retried      -> proceed to analyst anyway

Decision Point 3: Reviewer Routing
  IF review.passed = True               -> END (final_report set)
  IF review.passed = False
    AND revision_count < max_revisions  -> loop back to writer
    AND revision_count >= max_revisions -> END (force-pass, final_report set)

Decision Point 4: Budget Gate (every agent)
  IF token_count >= 50,000 -> skip LLM call, pass through state
  ELSE                     -> proceed normally
```

### End-to-End Data Flow

```
User Query (string)
     |
     v
[Planner] -> sub_topics: list[str]       (1-3 focused questions)
     |
     v
[Researcher x N] -> sources: list[dict]  (5-15 {title, url, snippet, tool} per topic)
     |                                     (merged via operator.add)
     v
[Quality Gate] -> quality_score: float    (0.0-1.0, mean of top 5 combined scores)
     |             quality_passed: bool    (score >= 0.4)
     |             source_ranking: list    (sorted by combined_score desc)
     v
[Analyst] -> key_claims: list[dict]       (5-8 {claim, source_idx, confidence, evidence})
     |        conflicts: list[dict]        (cross-source contradictions)
     v
[Synthesizer] -> synthesis: str           (3-5 paragraph narrative)
     |
     v
[Writer] -> current_draft: str            (structured report with ## sections)
     |       drafts: list[dict]            (versioned audit trail)
     |       revision_count: int           (incremented each write)
     v
[Reviewer] -> review: dict                ({score, issues, suggestions, passed})
     |         final_report: str           (= current_draft when passed)
     v
Streamlit UI / CLI Output
```

### Key Design Principles

1. **Single shared state** - All agents read from and write to the same `ResearchState`.
   No message passing, no agent-to-agent communication outside the state dict.

2. **operator.add for parallel merge** - Fields that accumulate across parallel `Send()`
   executions use `Annotated[..., operator.add]` for automatic merging.

3. **Budget-first checks** - Every agent checks `check_budget()` before making LLM calls.
   This prevents runaway costs even if the graph has bugs.

4. **Graceful degradation** - When budget is exceeded, agents skip their LLM call and
   pass through existing state. The pipeline produces the best report possible with
   the tokens it has.

5. **Bounded loops** - The writer-reviewer loop has a hard cap (`max_revisions=2`).
   The quality gate retry only fires once (`len(queries_used) > len(sub_topics)` check).

6. **Pure Python where possible** - The quality gate makes no LLM call. The researcher
   makes no LLM call. The tool selector uses keyword matching. This saves tokens for
   the agents that truly need reasoning (analyst, synthesizer, writer, reviewer).

7. **Structured output enforcement** - Three agents (planner, analyst, reviewer) use
   `with_structured_output(PydanticModel)` to guarantee JSON-valid responses that
   conform to the expected schema.

8. **Config-driven behavior** - All thresholds, limits, and model settings live in
   `configs/base.yaml`. Changing the quality threshold, max revisions, or even the
   LLM model requires zero code changes.

9. **Error isolation** - Each agent catches its own exceptions and appends to
   `state.errors`. A single agent failure does not crash the entire pipeline.

10. **Audit trail** - The `pipeline_trace` field records every agent's execution
    (duration, tokens, summary). The `drafts` list preserves every version of the
    report. Both are available in the UI for debugging.

---

*Document generated for the Multi-Agent Research System. 110 tests passing.*
