# Cost Analysis: Multi-Agent Research System

> Per-report cost breakdown using Gemini 3.6 Flash paid-tier pricing ($1.50/1M input, $7.50/1M output). A free tier is also available (Google AI Studio), under which these reports cost $0; the figures below are the paid-tier ceiling.

## Per-Agent Token Breakdown

> **Measured, not modelled.** Three live runs against `gemini-3.6-flash`, taking
> `usage_metadata` off each response rather than estimating. Web search ran
> against the offline stand-in, which does not change any agent's LLM tokens
> except through snippet length. Reproduce with `python run.py trace`.
>
> An earlier version of this table listed a "Fact-Checker" agent, which does not
> exist in this project, and its rows summed to 6,200 against a stated total of
> 5,200.

| Agent | Input | Output | Total | Share |
|-------|------:|-------:|------:|------:|
| Planner | 67 | 568 | 635 | 5% |
| Researcher x3 (Tavily) | 0* | 0* | 0* | 0% |
| Quality Gate | 0** | 0** | 0** | 0% |
| Analyst | 485 | 1,685 | 2,170 | 18% |
| Synthesizer | 790 | 1,847 | 2,637 | 22% |
| Writer | 1,144 | 3,284 | 4,428 | 36% |
| Reviewer | 1,063 | 1,300 | 2,363 | 19% |
| **Total** | **3,549** | **8,684** | **12,233** | **100%** |

*The researchers make no LLM call at all. They call Tavily and collect snippets,
which costs against the search quota but zero model tokens.

**The quality gate scores sources arithmetically (domain trust and snippet
length). No LLM call, which is exactly why it sits before the analyst.

Three things worth noticing.

**Output dominates.** 8,684 output tokens against 3,549 input, and output is
priced five times higher. Any cost estimate for this kind of pipeline that
reasons about prompt size is looking at the cheaper half.

**The writer is the most expensive agent**, at 36%, not the analyst. It is the
only agent that produces a long document, and it runs twice whenever the
reviewer rejects a draft.

**The previous figures were out by 3.2x on cost.** Input was almost exactly as
stated; output was understated by nearly four times. A plausible-looking table
that nobody had run is worse than no table.

Runs vary: the three measured totals were 11,294, 14,923 and 10,480 tokens. The
spread is the revision loop, which fired on one of the three.

## Cost Per Report

| Configuration | Tokens/Report | Cost/Report | Reports/$1 |
|--------------|--------------|-------------|------------|
| Single-agent baseline | 1,850 | $0.007 | ~140 |
| Multi-agent (7 agents, 8 nodes) | 12,233 | $0.071 | ~14 |
| Multi-agent + retry path | 14,923 | $0.086 | ~12 |

## Budget Enforcement

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Total budget | 50,000 tokens | ~$0.29/report max at the measured 71% output share |
| Per-agent limit | None (global only) | Simplicity; global cap prevents runaway |
| Graceful degradation | Yes | Produces best report at budget limit |
| Budget utilization | ~24.5% measured | Room for complex topics |

## Monthly Cost Projections

| Usage Level | Reports/Month | Monthly Cost |
|-------------|---------------|-------------|
| Light (dev/testing) | 100 | $7.00 |
| Moderate (production) | 1,000 | $70.00 |
| Heavy (enterprise) | 10,000 | $700.00 |

## Cost Optimization Strategies

1. **Route simple queries to single-agent** — saves 2.8x per factual lookup
2. **Cache Tavily results** — avoid duplicate searches for similar queries
3. **Truncate source snippets** — analyst only needs first 500 chars per source
4. **Skip fact-checker for low-stakes topics** — conditional pipeline edges
