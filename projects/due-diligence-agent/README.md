# Multi-Agent Due Diligence Analyst

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)

Enterprise-grade company research powered by **6 AI agents** with parallel execution, fact-checking, contradiction resolution, and comprehensive guardrails.

```
Input: "Tesla"  ──>  6 Agents (parallel)  ──>  Fact-Check  ──>  Debate  ──>  Final Report
                     Financial Analyst         Independent      Resolve       Risk Rating
                     News & Sentiment          Verification     Conflicts     Recommendation
                     Competitive Intel                                        Action Items
                     Risk Assessor
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/due-diligence-agent
pip install -r requirements.txt

# 2. Set API key (free at https://aistudio.google.com/apikey)
#    Option A: Create a .env file (recommended)
cp .env.example .env    # Then edit .env and paste your key

#    Option B: Set environment variable directly
#    Linux/Mac:   export GOOGLE_API_KEY=your_key_here
#    Windows CMD: set GOOGLE_API_KEY=your_key_here
#    PowerShell:  $env:GOOGLE_API_KEY='your_key_here'

# 3. Run analysis
python main.py --company "Tesla"

# Or launch the Streamlit dashboard
streamlit run app.py
```

> **Windows users:** If `pip install` fails with permission errors, use `pip install --user -r requirements.txt` or create a virtual environment first: `python -m venv .venv` then `.venv\Scripts\activate`.

## Architecture

```
                    +------------------+
                    |   Lead Analyst   |  PLAN: Decomposes query
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |              |
     +--------v--+  +-------v----+  +------v-------+  +--v-----------+
     | Financial  |  |   News &   |  | Competitive  |  |    Risk      |
     | Analyst    |  | Sentiment  |  | Intelligence |  |  Assessor    |
     +-----+------+  +-----+-----+  +------+-------+  +------+-------+
           |              |              |                    |
           +--------------+--------------+--------------------+
                             |
                    +--------v---------+
                    |   Fact Checker   |  VERIFY: Independent checks
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Lead Analyst   |  DEBATE: Resolve contradictions
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Lead Analyst   |  SYNTHESIZE: Final report
                    +------------------+
```

### What Makes This Enterprise-Grade

| Feature | Implementation |
|---------|---------------|
| **Parallel Execution** | 4 specialist agents via LangGraph `Send()` - 3x faster |
| **Structured Output** | Pydantic schemas via `.with_structured_output()` - zero parsing errors |
| **Graceful Degradation** | Every agent has try/except fallback - pipeline never halts |
| **Fact-Checking** | Independent verification pass challenges all claims |
| **Contradiction Resolution** | Lead Analyst debates conflicting findings |
| **Budget Guardrails** | Token ceiling, cost cap, loop detection, timeout enforcement |
| **PII Detection** | Email, phone, SSN, credit card pattern matching + masking |
| **Source Grounding** | Every claim must cite a URL - no unsourced assertions |
| **Search Caching** | SQLite-backed with TTL and LRU eviction |
| **Model Fallback** | Primary -> fallback model chain with exponential backoff |
| **Cost Tracking** | Per-agent token usage and cost estimation |

## Project Structure

```
due-diligence-agent/
├── configs/
│   └── base.yaml                 # All tunable parameters (single source of truth)
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph pipeline wiring + run_pipeline()
│   │   ├── lead_analyst.py       # Planner + Debater + Synthesizer (3 roles)
│   │   ├── financial_analyst.py  # Revenue, margins, ratios, cash flow
│   │   ├── news_sentiment.py     # News timeline, sentiment trends, PR patterns
│   │   ├── competitive_intel.py  # Competitors, moats, market position
│   │   ├── risk_assessor.py      # Legal, regulatory, operational, ESG risks
│   │   ├── fact_checker.py       # Independent claim verification
│   │   └── __main__.py           # CLI: python -m src.agents "Tesla"
│   ├── tools/
│   │   ├── search.py             # Web search (Tavily/DuckDuckGo) + SQLite cache
│   │   └── calculators.py        # Financial ratios, risk scoring, sentiment calc
│   ├── models/
│   │   ├── state.py              # DueDiligenceState TypedDict (shared state)
│   │   └── schemas.py            # 14 Pydantic schemas for structured LLM output
│   ├── guardrails/
│   │   └── manager.py            # Budget, loops, PII, source grounding, timeout
│   ├── config.py                 # YAML loader with env overrides + caching
│   └── llm.py                    # LLM factory: Google/OpenAI/Ollama + token tracking
├── tests/
│   ├── test_config.py            # Config loading, env overrides, defaults
│   ├── test_guardrails.py        # PII detection, budget, loops, disable
│   ├── test_calculators.py       # Financial ratios, risk scores, sentiment
│   ├── test_state.py             # State schema initialization
│   ├── test_graph.py             # Graph routing, compilation, guarded nodes
│   └── test_search.py            # Search cache CRUD, eviction
├── evaluation/
│   ├── run_eval.py               # Coverage, source diversity, consistency metrics
│   └── judge_prompt.py           # LLM-as-judge prompt templates
├── notebooks/
│   └── Due_Diligence_Agent.ipynb # Step-by-step walkthrough (Kaggle-ready)
├── docs/
│   └── architecture.md           # Detailed architecture docs
├── docker/
│   ├── Dockerfile                # Multi-stage build
│   └── docker-compose.yml        # One-command deployment
├── scripts/
│   └── run_pipeline.sh           # Shell script for CLI analysis
├── app.py                        # Streamlit dashboard
├── main.py                       # CLI entry point (analyze | ui | evaluate)
├── requirements.txt              # Pinned dependencies
├── Makefile                      # make test | analyze | ui | evaluate | clean
└── .env.example                  # API key template
```

## Usage

### CLI Analysis

```bash
# Standard analysis
python main.py --company "Tesla"

# Deep analysis with specific focus
python main.py --company "Stripe" --depth deep --query "Focus on fintech regulation risks"

# Quick scan with report output
python main.py --company "OpenAI" --depth quick --output reports/openai.md

# Via module
python -m src.agents "Databricks" --depth standard
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

Features:
- Company input with depth selection
- Real-time agent progress indicators
- Tabbed report viewer (Full Report / Findings / Trace / Raw Data)
- Budget and cost tracking
- Report download (Markdown)

### Docker

```bash
# Build and run
docker build -f docker/Dockerfile -t dd-agent .
docker run -p 8501:8501 -e GOOGLE_API_KEY=your_key dd-agent

# Or via docker-compose
cd docker && docker-compose up
```

### Python API

```python
from src.agents.graph import run_pipeline

result = run_pipeline("Tesla", depth="standard")

print(result["executive_summary"])
print(result["overall_risk_rating"])
print(result["final_report"])
```

## Run Tests

```bash
# All tests (64 tests)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
python -m pytest tests/test_guardrails.py -v

# If you have make installed (Linux/Mac)
make test
```

## Configuration

All parameters in `configs/base.yaml`. Override via environment variables:

| Env Variable | Config Path | Default |
|-------------|-------------|---------|
| `GOOGLE_API_KEY` | - | Required |
| `TAVILY_API_KEY` | - | Optional (falls back to DuckDuckGo) |
| `DD_MODEL_NAME` | model.name | gemini-2.5-flash |
| `DD_MAX_COST_USD` | budget.max_cost_usd | 0.50 |
| `DD_MAX_TOTAL_TOKENS` | budget.max_total_tokens | 100000 |
| `DD_LOG_LEVEL` | logging.level | INFO |

## Evaluation

```bash
python main.py --stage evaluate
```

Evaluates on 3 test companies (Tesla, Stripe, Anthropic) across 4 metrics:
- **Coverage**: Breadth of research areas (financial/news/competitive/risk)
- **Source diversity**: Number of unique sources cited
- **Factual consistency**: Cross-agent agreement (via fact-checker)
- **Actionability**: Verdict, risk rating, recommendations, uncertainty acknowledgment

## Cost

Using Google Gemini 2.5 Flash free tier (30 RPM, 1500 RPD):

| Depth | LLM Calls | Tokens | Cost | Duration |
|-------|-----------|--------|------|----------|
| Quick | 7 | ~20K | ~$0.008 | ~60s |
| Standard | 7-9 | ~28K | ~$0.012 | ~120s |
| Deep | 9-12 | ~40K | ~$0.018 | ~180s |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph, Send, conditional edges) |
| LLM | Google Gemini 2.5 Flash (free tier) |
| Structured Output | Pydantic v2 + `.with_structured_output()` |
| Search | Tavily (paid) / DuckDuckGo (free fallback) |
| Caching | SQLite (search results + TTL + LRU) |
| UI | Streamlit |
| Testing | pytest (64 tests) |
| Deployment | Docker multi-stage build |
| Config | YAML + environment variable overrides |

## Interview Guide

**Q: Walk me through the architecture. Why multiple agents instead of one big prompt?**

Single-prompt approaches hit context limits, can't parallelize, and produce monolithic outputs you can't debug. Our multi-agent design:
1. Enables parallel research (4 agents run concurrently)
2. Produces structured, auditable findings per domain
3. Allows independent fact-checking (an agent can't verify its own claims)
4. Supports graceful degradation (one agent failing doesn't kill the pipeline)
5. Scales - add a new specialist without changing existing agents

**Q: How do you handle hallucinations?**

Four layers:
1. **Source grounding**: Every agent prompt says "only state facts from search results, cite URLs"
2. **Structured output**: Pydantic schemas force agents to provide `sources` and `confidence` fields
3. **Fact Checker**: Independent agent re-searches key claims and flags contradictions
4. **Guardrails**: Post-execution check flags outputs without source URLs

**Q: What happens when an agent fails?**

Every agent has a three-layer fallback:
1. Primary LLM call with structured output
2. Retry with fallback model (exponential backoff)
3. Return degraded-but-valid output with error metadata

The pipeline ALWAYS produces a report, even if some agents fail. Failed agents contribute "Analysis incomplete - manual review recommended" findings instead of blocking the pipeline.

**Q: How do you prevent the system from running up a huge API bill?**

GuardrailManager enforces five budgets:
- Token ceiling (default: 100K tokens)
- Cost ceiling (default: $0.50)
- Agent call limit (default: 30 calls)
- Per-agent timeout (default: 120s)
- Pipeline timeout (default: 300s)

Pre-execution checks block agents that would exceed any limit. The system degrades gracefully with partial results rather than burning through budget.

**Q: What would you improve for production?**

1. Add async execution (currently synchronous within each agent)
2. Add persistent memory (cross-analysis learning)
3. Add a human-in-the-loop approval gate before the final report
4. Add real-time WebSocket streaming for the dashboard
5. Add RAG over proprietary data sources (SEC filings, internal databases)
6. Add multi-language support for global companies
7. Add scheduled re-analysis with drift detection

## License

MIT
