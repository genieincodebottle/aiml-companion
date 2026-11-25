# Architecture - Multi-Agent Due Diligence Analyst

## System Overview

```
                         +-------------------+
                         |   User Input      |
                         |   (Company Name)  |
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Lead Analyst    |  Phase 1: PLAN
                         |   (Planner)       |  Decomposes query into
                         |                   |  agent-specific tasks
                         +--------+----------+
                                  |
                    +-------------+-------------+
                    |             |             |             |
           +--------v--+  +------v-----+  +----v-------+  +--v-----------+
           | Financial  |  |   News &   |  | Competitive|  |    Risk      |
           | Analyst    |  | Sentiment  |  | Intel      |  |  Assessor    |
           |            |  |            |  |            |  |              |
           | - Revenue  |  | - Timeline |  | - Market   |  | - Legal      |
           | - Margins  |  | - PR       |  | - Moats    |  | - Regulatory |
           | - Ratios   |  | - Social   |  | - Threats  |  | - ESG        |
           +-----+------+  +-----+-----+  +-----+------+  +------+------+
                 |              |              |                   |
                 +--------------+--------------+-------------------+
                                  |
                         +--------v----------+
                         |   Fact Checker    |  Phase 2: VERIFY
                         |                   |  Independent verification
                         |   - Verify claims |  of high-priority claims
                         |   - Flag conflicts|
                         +--------+----------+
                                  |
                         +--------v----------+
                         |   Lead Analyst    |  Phase 3: DEBATE
                         |   (Debater)       |  Resolves contradictions
                         +--------+----------+  between agents
                                  |
                         +--------v----------+
                         |   Lead Analyst    |  Phase 4: SYNTHESIZE
                         |   (Synthesizer)   |  Final report with
                         |                   |  executive summary
                         +-------------------+
```

## Key Architectural Decisions

### 1. Parallel Execution via LangGraph Send()
Specialist agents run concurrently, not sequentially, so the run waits for the slowest
specialist rather than the sum of all four.
LangGraph's `Send()` mechanism handles fan-out and automatic state merging.

### 2. Structured Output (Zero Parsing Errors)
Every LLM call uses `.with_structured_output(PydanticModel)`. The LLM is constrained
to return valid JSON matching the schema. No regex parsing, no JSON.parse failures.

### 3. Append-Only State (No Lost Updates)
List fields use `Annotated[list, operator.add]` so parallel agents' outputs are
merged (appended), not overwritten. This prevents the last-writer-wins problem.

### 4. Three-Layer Fallback
```
Layer 1: LLM call with structured output
    |
    v (fails)
Layer 2: Retry the same model, up to pipeline.max_retries_per_agent, with backoff
    |
    v (fails)
Layer 3: Return degraded-but-valid output with error metadata
```

`model.fallback_name` is a separate mechanism: `get_llm()` switches to it only when
creating the primary client fails (for example a bad model name). A call that fails after
the client exists is retried on the same model, never on the fallback.

Every agent ALWAYS returns valid state, even on total failure. The pipeline never
halts due to a single agent error.

### 5. Guardrail-Wrapped Execution
Every agent call passes through pre/post guardrail checks:
- **Pre**: Token budget, per-agent token ceiling, cost ceiling, loop detection (call count and repeat entry with an identical state), timeout
- **Run**: per-agent timeout (`pipeline.timeout_per_agent_seconds`), per-agent token ceiling enforced before each LLM call, LLM retries capped by `pipeline.max_retries_per_agent`
- **Post**: charge real token/cost usage to the budgets, PII masking, source grounding, confidence thresholding (low-confidence findings are flagged for review, not dropped)

### 6. Fact-Checking as Architecture
The Fact Checker is a core pipeline stage and is on by default. It can be switched off
with `guardrails.hallucination_check_enabled: false`; the node then adds a warning and
every finding stays UNVERIFIED. When it runs, it provides:
- Independent verification (separate search queries)
- Cross-agent contradiction detection
- Confidence calibration
- Trust signal for downstream consumers

## Data Flow

```
Input: company_name, query, depth
  |
  v
plan_research() -> research_plan, focus_areas
  |
  v (parallel)
financial_analyst() -> financial_findings[]
news_sentiment()    -> news_findings[]
competitive_intel() -> competitive_findings[]
risk_assessor()     -> risk_findings[]
  |
  v (converge)
fact_checker() -> fact_check_results[], contradictions[]
  |
  v (conditional)
resolve_contradictions() -> debate_log[] (only if contradictions exist)
  |
  v
synthesize_report() -> executive_summary, final_report, risk_rating
```

## File Structure

```
due-diligence-agent/
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph wiring + run_pipeline()
│   │   ├── lead_analyst.py       # Plan + Debate + Synthesize (3 roles)
│   │   ├── financial_analyst.py  # Financial health research
│   │   ├── news_sentiment.py     # News & sentiment analysis
│   │   ├── competitive_intel.py  # Market & competitive landscape
│   │   ├── risk_assessor.py      # Multi-dimensional risk scoring
│   │   ├── fact_checker.py       # Independent claim verification
│   │   └── __main__.py           # CLI entry point
│   ├── tools/
│   │   ├── search.py             # Web search with caching + fallback
│   │   └── calculators.py        # Financial ratios, risk scoring
│   ├── models/
│   │   ├── state.py              # DueDiligenceState TypedDict
│   │   └── schemas.py            # Pydantic structured output schemas
│   ├── guardrails/
│   │   └── manager.py            # Budget, loops, PII, source grounding
│   ├── config.py                 # YAML config loader with env overrides
│   └── llm.py                    # LLM factory with fallback + token tracking
├── tests/                        # 100+ pytest tests
├── evaluation/                   # Coverage, diversity, consistency metrics
├── configs/base.yaml             # All tunable parameters
├── app.py                        # Streamlit dashboard
├── main.py                       # CLI entry point
└── docker/                       # Dockerfile + docker-compose
```

## Model Calls per Run

| Pipeline Stage | LLM Calls |
|---------------|-----------|
| Planning | 1 |
| 4 Specialists | 4 |
| Fact Checker | 1 (0 when disabled) |
| Debate (if needed) | 0-2 |
| Synthesis | 1 |
| **Total** | **7-9** |

Token use depends on the company, the depth and the search results, so there is no fixed
figure. Each run reports the tokens it actually used, per node, in `pipeline_trace` and in
`_token_summary` on the final state. The budget ceilings (`budget.max_total_tokens`,
`budget.max_tokens_per_agent`) are read from `configs/base.yaml`.
