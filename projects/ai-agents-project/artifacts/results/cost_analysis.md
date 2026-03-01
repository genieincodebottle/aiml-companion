# Cost Analysis: Multi-Agent Research System

> Per-report cost breakdown using gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output)

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
| Single-agent baseline | 1,850 | $0.0003 | ~3,300 |
| Multi-agent (4 agents) | 5,200 | $0.0008 | ~1,250 |
| Multi-agent + retries | 7,800 | $0.0012 | ~830 |

## Budget Enforcement

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total budget | 50,000 tokens | ~$0.008/report max |
| Per-agent limit | None (global only) | Simplicity; global cap prevents runaway |
| Graceful degradation | Yes | Produces best report at budget limit |
| Budget utilization | ~10% typical | Room for complex topics |

## Monthly Cost Projections

| Usage Level | Reports/Month | Monthly Cost |
|-------------|---------------|-------------|
| Light (dev/testing) | 100 | $0.08 |
| Moderate (production) | 1,000 | $0.80 |
| Heavy (enterprise) | 10,000 | $8.00 |

## Cost Optimization Strategies

1. **Route simple queries to single-agent** — saves 2.8x per factual lookup
2. **Cache Tavily results** — avoid duplicate searches for similar queries
3. **Truncate source snippets** — analyst only needs first 500 chars per source
4. **Skip fact-checker for low-stakes topics** — conditional pipeline edges
