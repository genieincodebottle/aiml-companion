# Cost Analysis: Multi-Agent Research System

> Per-report cost breakdown using Gemini 3.6 Flash paid-tier pricing ($1.50/1M input, $7.50/1M output). A free tier is also available (Google AI Studio), under which these reports cost $0; the figures below are the paid-tier ceiling.

## Per-Agent Token Breakdown

| Agent | Avg Input Tokens | Avg Output Tokens | Avg Total | % of Budget |
|-------|-----------------|-------------------|-----------|-------------|
| Researcher (Tavily) | 50 | 0 | ~500* | 10% |
| Analyst | 1,200 | 800 | 2,000 | 38% |
| Writer | 1,500 | 1,000 | 2,500 | 48% |
| Fact-Checker | 800 | 400 | 1,200 | 23% |
| **Total** | **3,550** | **2,200** | **~5,200** | **10.4%** |

*Researcher tokens estimated (Tavily API call, not LLM tokens)

## Cost Per Report

| Configuration | Tokens/Report | Cost/Report | Reports/$1 |
|--------------|--------------|-------------|------------|
| Single-agent baseline | 1,850 | $0.007 | ~140 |
| Multi-agent (4 agents) | 5,200 | $0.020 | ~50 |
| Multi-agent + retries | 7,800 | $0.030 | ~33 |

## Budget Enforcement

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total budget | 50,000 tokens | ~$0.19/report max |
| Per-agent limit | None (global only) | Simplicity; global cap prevents runaway |
| Graceful degradation | Yes | Produces best report at budget limit |
| Budget utilization | ~10% typical | Room for complex topics |

## Monthly Cost Projections

| Usage Level | Reports/Month | Monthly Cost |
|-------------|---------------|-------------|
| Light (dev/testing) | 100 | $2.00 |
| Moderate (production) | 1,000 | $20.00 |
| Heavy (enterprise) | 10,000 | $200.00 |

## Cost Optimization Strategies

1. **Route simple queries to single-agent** — saves 2.8x per factual lookup
2. **Cache Tavily results** — avoid duplicate searches for similar queries
3. **Truncate source snippets** — analyst only needs first 500 chars per source
4. **Skip fact-checker for low-stakes topics** — conditional pipeline edges
