# Multi-Agent Research System

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Tests](https://img.shields.io/badge/tests-110%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 7-agent orchestrated research pipeline (8 graph nodes) with quality-gated routing, parallel fan-out, iterative refinement, and production guardrails

## Problem Statement

Single LLM calls produce mediocre research reports: they hallucinate citations, miss key perspectives, and have no quality verification. This project builds a **production-grade multi-agent system** where 7 specialized agent functions collaborate across 8 graph nodes through shared state, quality-gated routing, and iterative refinement to produce verified, well-cited research reports.

## Architecture

```
Research Topic (user input)
    |
    v
+------------------------------------------+
|  Guardrails Layer                         |
|  PII Scrub | URL Validate | Budget Cap   |
|  Prompt Injection Detection              |
|  Rate Limiter (30 RPM Gemini limit)      |
+------------------------------------------+
    |
    v
+-----------+
|  Planner  |  Decompose query into 1-3 sub-topics
+-----------+  with_structured_output(PlannerOutput)
    |
    | Send() fan-out (parallel)
    v
+-----------+  +-----------+  +-----------+
| Researcher|  | Researcher|  | Researcher|  Multi-tool: Tavily + scraper + Wikipedia
| (topic 1) |  | (topic 2) |  | (topic 3) |  operator.add merges sources
+-----------+  +-----------+  +-----------+
    |               |               |
    +-------+-------+-------+-------+
            |
            v
    +---------------+
    | Quality Gate  |  Pure Python: domain trust + snippet scoring
    +---------------+  No LLM call (saves budget)
            |
     +------+------+
     |             |
  score < 0.4   score >= 0.4
     |             |
     v             v
+-----------+  +-----------+
| Researcher|  |  Analyst  |  Extract claims + confidence + evidence
| (retry)   |  +-----------+
+-----------+       |
                    v
             +-----------+
             |Synthesizer|  Cross-reference, detect conflicts, rank sources
             +-----------+
                    |
                    v
             +-----------+
             |  Writer   |  Versioned drafts with citations
             +-----------+
                    |
                    v
             +-----------+
             | Reviewer  |  Score 0-10, flag issues
             +-----------+
                    |
             +------+------+
             |             |
          score < 7     score >= 7
          & rev < 2        |
             |             v
             v          +-----+
         +-----------+  | END |
         |  Writer   |  +-----+
         | (revision)|
         +-----------+
```

## Key LangGraph Patterns Demonstrated

| Pattern | Where | What It Teaches |
|---------|-------|-----------------|
| `Send()` parallel fan-out | Planner -> Researchers | Parallel agent execution with state merging |
| `operator.add` reducer | `sources` field | Merging results from parallel nodes |
| Conditional edge (pure Python) | Quality Gate | Routing without LLM calls |
| Iterative refinement loop | Writer <-> Reviewer | Bounded loops with max revision guard |
| `with_structured_output()` | Planner, Analyst, Reviewer | Pydantic-validated agent outputs |
| YAML-driven config | `configs/base.yaml` | Config-driven agent registry |
| SQLite research cache | `src/cache/` | Source dedup and query caching |
| Rate limiting | `src/guardrails.py` | Respecting API rate limits (Gemini 30 RPM) |

## Results

### Single-Agent vs Multi-Agent Comparison

| Metric | Single-Agent | Multi-Agent | Improvement |
|--------|-------------|-------------|-------------|
| Accuracy (0-3) | 2.2 | 2.6 | +18% |
| Completeness (0-3) | 1.8 | 2.4 | +33% |
| Citations (0-3) | 1.4 | 2.0 | +43% |
| Avg Cost/Report | $0.0003 | $0.0008 | 2.8x |

**Verdict:** Multi-agent justified for synthesis tasks (+33% completeness). Route simple factual lookups to single-agent to save cost.

---

## Setup

> **100% free to run.** This project uses Google Gemini (free tier) and Tavily Search (free tier). No credit card, no payment, no trial period. You just need two free API keys (takes 2 minutes to get both).

```bash
# 1. Create virtual environment
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Set up API keys (both FREE - no credit card needed)
cp .env.example .env
# Edit .env with your keys:
#   GOOGLE_API_KEY=your_gemini_api_key    (free from https://aistudio.google.com/apikey)
#   TAVILY_API_KEY=your_tavily_api_key    (free from https://app.tavily.com)
```

### API Keys Required (Both Free)

| Key | Purpose | Free Tier | Get It Here |
|-----|---------|-----------|-------------|
| `GOOGLE_API_KEY` | Gemini LLM (all agent calls) | 30 RPM, 1500 RPD | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `TAVILY_API_KEY` | Web search tool | 1000 searches/month | [app.tavily.com](https://app.tavily.com/home) |

**How to get your keys (2 minutes):**
1. **Gemini key**: Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey), sign in with your Google account, click "Create API key", copy it
2. **Tavily key**: Go to [app.tavily.com](https://app.tavily.com/home), sign up with email or GitHub, your API key appears on the dashboard, copy it
3. Paste both keys into your `.env` file (replace the placeholder values)

> **No credit card required.** Both keys are completely free. The Gemini free tier allows 30 requests per minute and 1500 per day, which is more than enough for this project.

---

## Testing Guide

This project has **110 automated tests** across 8 test files covering unit tests, integration tests, end-to-end pipeline tests, and UI smoke tests.

### Quick Start: Run All Tests

```bash
# Run all 110 tests (no API keys needed)
make test

# Or directly with pytest
pytest tests/ -v
```

**Expected output:**
```
tests/test_cache.py          5 passed    - SQLite cache operations
tests/test_e2e_pipeline.py  37 passed    - Full pipeline with mocked LLM
tests/test_graph.py          9 passed    - Graph wiring and routing
tests/test_guardrails.py    14 passed    - PII, budget, rate limiting
tests/test_quality_gate.py   8 passed    - Source quality scoring
tests/test_state.py          6 passed    - State schema validation
tests/test_tools.py          5 passed    - Tool selector logic
tests/test_ui_smoke.py      26 passed    - Streamlit UI imports
───────────────────────────────────────
                           110 passed
```

### Test Categories Explained

#### 1. Unit Tests (38 tests) - `test_guardrails.py`, `test_quality_gate.py`, `test_state.py`, `test_tools.py`, `test_cache.py`

Test individual components in isolation. No LLM calls, no network.

```bash
# Run only unit tests
pytest tests/test_guardrails.py tests/test_quality_gate.py tests/test_state.py tests/test_tools.py tests/test_cache.py -v
```

**What they verify:**
- PII detection catches emails, phones, SSNs
- PII scrubbing replaces sensitive data with `[REDACTED]`
- Budget enforcement blocks at 50K token limit
- Rate limiter tracks API calls per window
- Quality gate scores sources by domain trust + snippet quality
- Default state schema has all 18+ required fields
- Pydantic schemas validate planner, reviewer, and claim outputs
- Tool selector routes queries to correct tools (Tavily, Wikipedia, scraper)
- SQLite cache stores/retrieves sources with case-insensitive matching

#### 2. Graph Wiring Tests (9 tests) - `test_graph.py`

Test LangGraph routing logic without running the actual pipeline.

```bash
pytest tests/test_graph.py -v
```

**What they verify:**
- Graph compiles successfully with all nodes
- Quality gate routes to `analyst` when `quality_passed=True`
- Quality gate routes to `retry_researcher` when `quality_passed=False`
- Quality gate routes to `analyst` if already retried (prevents infinite loop)
- Reviewer routes to `END` when score >= 7
- Reviewer routes to `writer` when score < 7 and revision_count < 2
- Reviewer routes to `END` when max revisions reached
- `Send()` fan-out creates one researcher per sub-topic
- YAML config loads correctly

#### 3. End-to-End Pipeline Tests (37 tests) - `test_e2e_pipeline.py`

Test the full pipeline flow with **mocked LLM responses**. These tests verify the complete orchestration without making real API calls.

```bash
pytest tests/test_e2e_pipeline.py -v
```

**What they verify:**

| Test Class | Tests | What It Checks |
|-----------|-------|----------------|
| `TestFullPipelineMocked` | 2 | Full happy-path pipeline, agent execution order in trace |
| `TestQualityGateRetryFlow` | 5 | Bad sources trigger retry, good sources pass, routing logic |
| `TestReviewerRefinementLoop` | 4 | Score >= 7 ends pipeline, score < 7 loops, max revisions cap |
| `TestParallelFanOut` | 4 | Send() creates per-subtopic researchers, passes token count |
| `TestBudgetEnforcement` | 4 | Budget OK under limit, exceeded detection, custom budgets |
| `TestPIIScrubbing` | 5 | Email/phone/SSN scrubbing, clean text unchanged |
| `TestCacheIntegration` | 2 | Cache roundtrip, stats update after write |
| `TestStateSchemaE2E` | 3 | Default state fields, types, graph compiles with schema |
| `TestScoringHeuristics` | 6 | Domain trust scores, snippet quality assessment |
| `TestRateLimiterE2E` | 2 | Call tracking, reset functionality |

**Example: Full Pipeline Happy Path**

The test mocks all LLM calls and verifies the pipeline flows through all agents:

```python
# Planner decomposes query -> 2 sub-topics
# Researcher runs in parallel (via Send) for each sub-topic
# Quality gate scores sources -> passes (score >= 0.4)
# Analyst extracts claims with evidence
# Synthesizer cross-references and ranks sources
# Writer produces versioned draft
# Reviewer scores draft -> passes (score >= 7)
# Pipeline ends with final_report set
```

#### 4. UI Smoke Tests (26 tests) - `test_ui_smoke.py`

Test that the Streamlit app can be imported and all modules are wired correctly. Does NOT launch a Streamlit server.

```bash
pytest tests/test_ui_smoke.py -v
```

**What they verify:**
- All agent modules import without errors
- All 4 tool modules import without errors
- Cache module exports correct functions and returns valid stats
- Config module loads YAML and returns expected keys
- Guardrail module exports all functions (PII, budget, rate limiter)
- Evaluation module imports test questions and scoring functions
- Judge prompts contain required scoring criteria keywords

### Testing Without API Keys

All 110 tests run **without any API keys**. The e2e tests use `unittest.mock.patch` to mock LLM calls, so you can verify the full pipeline logic locally.

---

## Running the Pipeline (Interactive)

### Option 1: Streamlit UI (Recommended for Interactive Use)

```bash
make ui
# Or: streamlit run app.py
```

This launches a web interface where you can:
- Enter a research topic
- Watch the pipeline execute agent-by-agent with live status updates
- See the graph topology (which agent is currently active)
- View the pipeline trace (agent, duration, token usage per step)
- Read the final structured research report
- Monitor cache hits/misses and budget usage

### Option 2: Command Line

```bash
make run
# Or: python -m src.agents
```

Runs the pipeline directly and prints the final report to stdout. Uses the default query from the config.

### Option 3: Run Evaluation

```bash
make evaluate
# Or: python -m evaluation.run_eval
```

Runs 10 test questions through both single-agent and multi-agent pipelines. Produces a head-to-head comparison using LLM-as-judge scoring on accuracy, completeness, and citation quality.

### Option 4: Run Everything

```bash
make all
# Runs: tests -> pipeline -> evaluation
```

---

## Project Structure

```
ai-agents-project/
├── src/
│   ├── agents/                     # 7 agent functions (8 graph nodes)
│   │   ├── planner.py              # Query decomposition (structured output)
│   │   ├── researcher.py           # Multi-tool research (Tavily + scrape + wiki)
│   │   ├── quality_gate.py         # Pure Python source scoring (no LLM)
│   │   ├── analyst.py              # Claim extraction with evidence linking
│   │   ├── synthesizer.py          # Cross-source conflict detection
│   │   ├── writer.py               # Versioned report generation
│   │   ├── reviewer.py             # Draft review + scoring
│   │   └── graph.py                # LangGraph StateGraph wiring
│   ├── tools/                      # Research tools
│   │   ├── search.py               # Tavily web search wrapper
│   │   ├── scraper.py              # BeautifulSoup web scraper
│   │   ├── wikipedia.py            # Wikipedia API search
│   │   └── tool_selector.py        # Query-type -> tool routing
│   ├── cache/
│   │   └── research_cache.py       # SQLite source/query cache (24h TTL)
│   ├── models/
│   │   └── state.py                # ResearchState TypedDict + Pydantic schemas
│   ├── config.py                   # YAML config loader
│   └── guardrails.py               # PII detection, budget cap, rate limiter, prompt injection detection
├── tests/                          # 110 tests across 8 files
│   ├── test_e2e_pipeline.py        # Full pipeline e2e (37 tests, mocked LLM)
│   ├── test_ui_smoke.py            # Streamlit UI imports (26 tests)
│   ├── test_guardrails.py          # Guardrails unit tests (14 tests)
│   ├── test_graph.py               # Graph routing tests (9 tests)
│   ├── test_quality_gate.py        # Quality scoring tests (8 tests)
│   ├── test_state.py               # State schema tests (6 tests)
│   ├── test_cache.py               # Cache operation tests (5 tests)
│   └── test_tools.py               # Tool selector tests (5 tests)
├── evaluation/
│   ├── run_eval.py                 # Single vs multi-agent comparison
│   └── judge_prompt.py             # LLM-as-judge scoring prompts
├── configs/
│   └── base.yaml                   # Agent registry + pipeline config
├── artifacts/results/
│   ├── sample_report.md            # Example research report
│   ├── evaluation_results.md       # Comparison table
│   └── cost_analysis.md            # Token usage breakdown
├── scripts/
│   ├── initialize_cache.py         # Seed SQLite cache with sample queries
│   ├── cleanup_data.py             # Reset cache and clear __pycache__
│   ├── run_agents.sh               # Shell wrapper to run pipeline
│   └── run_evaluation.sh           # Shell wrapper to run evaluation
├── app.py                          # Streamlit UI (graph viz, trace, streaming)
├── Makefile                        # Build targets: run, test, ui, evaluate, all
├── requirements.txt                # Dependencies (12 packages)
├── END_TO_END_ARCHITECTURE.md      # 2400+ line architecture deep-dive
└── README.md
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Orchestration | LangGraph StateGraph | Explicit state flow, debuggable edges, built-in persistence |
| Agent count | 7 functions across 8 graph nodes (planner, researcher, quality_gate, analyst, synthesizer, writer, reviewer) - retry_researcher reuses the researcher function | Each handles a distinct failure mode. Quality gate and reviewer add routing intelligence |
| Parallel execution | `Send()` fan-out | Sub-topics researched concurrently, results merged via `operator.add` |
| Quality routing | Pure Python quality gate | No LLM call needed, routes by domain trust + snippet heuristics |
| Iterative refinement | Writer <-> Reviewer loop (max 2 revisions) | Catches issues without infinite loops |
| Search tools | Tavily + Wikipedia + scraper | Different tools for different query types (auto-selected) |
| Research cache | SQLite with 24h TTL | Prevents duplicate API calls, tracks cache statistics |
| LLM | Gemini (Google) | Free tier (1500 requests/day), structured output support |
| Budget enforcement | Global 50K token cap | Prevents runaway costs; agents degrade gracefully at limit |
| Rate limiting | 30 RPM limiter | Respects Gemini free tier limits |
| Evaluation | LLM-as-judge (Gemini) | Scalable, reproducible scoring on accuracy/completeness/citations |
| PII handling | Regex detection + scrubbing | Fast, no external dependencies, catches common PII patterns |
| Config | YAML-driven | Change model, thresholds, and agent behavior without code changes |
| UI | Streamlit | Rapid prototyping, built-in streaming, no frontend build step |

## Interview Guide

**Q: Why 7 agent functions (8 graph nodes) instead of 4?**
> The original 4-agent chain had no quality control or iteration. Adding a quality gate (pure Python, no LLM cost) enables quality-based routing - bad sources get retried before analysis. The retry_researcher node reuses the same researcher function, so there are 7 unique agent functions across 8 graph nodes. The reviewer loop ensures the final report meets a minimum quality bar. The planner enables parallel research via Send() fan-out.

**Q: How does parallel execution work in LangGraph?**
> The planner decomposes the query into 1-3 sub-topics. The `route_to_researchers` function creates `Send("researcher", state_for_topic)` objects, one per sub-topic. LangGraph executes these in parallel. The `sources` field uses `Annotated[list, operator.add]` to automatically merge results from all parallel researchers.

**Q: What's the quality gate and why is it pure Python?**
> It scores sources using domain trust (arxiv.org scores 0.95, reddit.com scores 0.3) and snippet quality (length, presence of data patterns like numbers/percentages). No LLM call needed - saves budget for the agents that truly need reasoning. If score < 0.4, it routes to a retry researcher for more sources.

**Q: How do you prevent the refinement loop from running forever?**
> Two guards: (1) the reviewer must score < 7/10 to trigger a revision, and (2) `revision_count` is capped at 2. The conditional edge checks both conditions: `score < 7 AND revision_count < max_revisions`. After 2 revisions, the pipeline ends regardless of score.

**Q: How do you prevent agents from hallucinating citations?**
> Three layers: (1) Writer constrained to cite ONLY from `state.sources`, (2) Analyst extracts claims with explicit evidence links to source indices, (3) Synthesizer detects cross-source conflicts and ranks sources by reliability.

**Q: How do you control costs in a multi-agent system?**
> Four mechanisms: (1) Global 50K token budget tracked in shared state, (2) Rate limiter caps at 30 requests/minute for Gemini free tier, (3) SQLite cache prevents duplicate searches (24h TTL), (4) Quality gate is pure Python (zero LLM cost). Worst case: 10 LLM calls per query at 2-second spacing.

**Q: What would you change for production deployment?**
> (1) Swap SQLite cache for Redis for concurrent access, (2) Add human-in-the-loop for low-confidence claims, (3) Implement streaming output via LangGraph's `astream_events`, (4) Add LangSmith tracing for observability, (5) Route simple queries to single-agent to save 2.8x cost.

**Q: How are the tests structured?**
> 110 tests in 4 categories: (1) Unit tests for individual components (guardrails, quality gate, state, tools, cache), (2) Graph wiring tests verify routing logic without running the pipeline, (3) E2E tests mock all LLM calls and run the full pipeline to verify orchestration, (4) UI smoke tests verify all Streamlit imports work. All tests run without API keys.

## Architecture Deep-Dive

For a comprehensive 2400+ line walkthrough of the entire system, see [END_TO_END_ARCHITECTURE.md](END_TO_END_ARCHITECTURE.md). It covers:
- Complete request flow (happy path, quality gate retry, reviewer refinement)
- Sequence diagrams for all 3 flow paths
- State schema with all 18+ fields and reducer annotations
- Database schema (SQLite cache tables)
- Performance metrics, security, and configuration guide
- Full testing architecture (all 110 tests explained)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Send() API](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [Tavily Search API](https://tavily.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
