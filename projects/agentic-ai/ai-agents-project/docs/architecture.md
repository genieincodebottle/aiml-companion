# Architecture: Multi-Agent Research System

This document describes the system architecture of the Multi-Agent Research pipeline:
the eight-node graph, the shared state, and the guardrails. The diagram is
[high-level-architecture.svg](high-level-architecture.svg); the full module-by-module
reference is [END_TO_END_ARCHITECTURE.md](../END_TO_END_ARCHITECTURE.md).

## Overview

The system uses a **LangGraph StateGraph** (`src/agents/graph.py`) with seven agent
functions on eight nodes (`retry_researcher` reuses the researcher function). Agents
communicate only through a shared typed state dictionary. Two edges are conditional, so
the graph can go backwards: a weak source set gets one more search, and a rejected draft
goes back to the writer.

## Pipeline Flow

```
                             User query
                                 |
                                 v
                    planner (screen_input, 1 LLM call)
                                 |
                  Send() one per sub-topic (1 to 3)
                 /               |               \
          researcher        researcher        researcher      search only, no LLM
                 \               |               /
                  sources merged by operator.add
                                 |
                                 v
                quality_gate (pure Python, threshold 0.4)
                    |                          |
              score >= 0.4               score < 0.4 (once)
                    |                          v
                    |                   retry_researcher
                    |                          |
                    v                          v
                         analyst (1 LLM call)
                                 |
                                 v
                       synthesizer (1 LLM call)
                                 |
                                 v
                 +------> writer (1 LLM call, PII scrubbed)
                 |               |
                 |               v
                 |       reviewer (1 LLM call)
                 |               |
                 +-- not passed, first draft
                                 |
                       passed, or second draft
                                 v
                           END: final_report
```

## Agent Responsibilities

### 1. Planner (`src/agents/planner.py`)

- **Input:** `query`
- **Output:** `sub_topics` (1 to 3, capped by `pipeline.max_sub_topics`), `research_plan`
- **Guardrails:** `screen_input()` on the query first; with `action: block` a match ends
  the run with the reason in `errors`. Then `check_budget()`.
- **Error handling:** if the LLM call fails, the whole query becomes the only sub-topic.

### 2. Researcher (`src/agents/researcher.py`)

- **Input:** one sub-topic, dispatched by `Send()`; the instances run in parallel
- **Output:** `sources[]` (title, url, snippet, date, tool), merged by `operator.add`
- **Tools:** `select_tools()` keyword rules: Tavily for every sub-topic, Wikipedia as well
  when the sub-topic contains a factual keyword ("what is", "history of", ...).
  Without a real `TAVILY_API_KEY`, `web_search()` returns the offline stand-in sources,
  and the writer labels them as placeholders in the prompt and in the report. In offline
  mode `wiki_search()` returns nothing, so there are no network calls.
- **Guardrail:** every fetched snippet goes through `screen_sources()` (prompt injection)
  before the merge.
- **No LLM call.** It reports 0 tokens. `throttle()` runs before each search.

### 3. Quality Gate (`src/agents/quality_gate.py`)

- Pure Python. Each source scores `0.6 x domain trust + 0.4 x snippet quality`; the gate
  averages the top five and passes at `pipeline.quality_threshold` (0.4).
- Below the threshold the run goes to `retry_researcher` once, which searches
  `"<query> comprehensive overview analysis"` and then goes straight to the analyst.

### 4. Analyst (`src/agents/analyst.py`)

- **Input:** the first 10 sources
- **Output:** `key_claims[]` (claim, source index, confidence, evidence) and `conflicts`
- One structured call (`AnalystOutput`).

### 5. Synthesizer (`src/agents/synthesizer.py`)

- **Input:** claims and conflicts
- **Output:** `synthesis` (free text: cross-references and source reliability)

### 6. Writer (`src/agents/writer.py`)

- **First call:** drafts the report from the claims and the synthesis.
- **Later call:** revises the last draft against the reviewer's issues.
- `guardrails.pii_detection.action` is applied to every draft: `redact` replaces emails,
  phone numbers and SSNs with `[REDACTED_TYPE]`, `warn` logs them, `block` drops the draft. Each draft is kept
  in `drafts` with a version number; `revision_count` holds that version number.

### 7. Reviewer (`src/agents/reviewer.py`)

- One structured call (`ReviewOutput`): score 1 to 10, issues, suggestions, `passed`.
- `passed` sets `final_report`. When `revision_count` reaches `pipeline.max_revisions`
  (2), the draft is force-passed, so a run writes at most two drafts (one revision).
- The pass is decided in code: `score >= pipeline.review_pass_score` (7). The model's own
  `passed` flag is ignored. `max_revisions` counts drafts, not revisions: 2 means a first
  draft and one revision, and the second draft is accepted whatever its score.
- `final_report` is cut at `guardrails.content_filtering.max_report_chars`.

## Shared State Design

All agents read from and write to one `ResearchState` TypedDict (`src/models/state.py`,
19 fields). The ones that matter most:

```python
class ResearchState(TypedDict):
    query: str
    sub_topics: list[str]
    sources: Annotated[list[dict], operator.add]          # parallel researchers append
    search_queries_used: Annotated[list[str], operator.add]
    quality_score: float
    quality_passed: bool
    key_claims: list[dict]
    conflicts: list[dict]
    synthesis: str
    drafts: list[dict]
    current_draft: str
    review: dict
    revision_count: int
    token_count: Annotated[int, operator.add]             # every agent adds its real usage
    errors: Annotated[list[str], operator.add]
    final_report: str
    pipeline_trace: Annotated[list[dict], operator.add]
```

Key design decisions for shared state:

- **Reducers for parallel writes:** fields written by the parallel researchers use
  `operator.add`, so each branch's results are appended, not overwritten.
- **Real token counts:** `src/token_usage.py` reads `usage_metadata` from each response.
  A response without it counts as 0 and logs a warning; there are no per-agent constants.
- **Append-only error log:** agents append to `errors` rather than overwriting.
- **Graceful degradation:** when the budget is spent, an agent skips its LLM call and
  passes the state on; the reviewer accepts the current draft as the report.

## Guardrails Layer

The guardrails module (`src/guardrails.py`) provides these safety mechanisms:

### PII Detection and Scrubbing

- Compiled regex patterns for email, phone, and SSN
- `detect_pii(text)` returns matched PII by type
- `scrub_pii(text)` replaces PII with `[REDACTED_TYPE]` tokens
- The writer calls `apply_pii_policy()` on every draft (`pii_detection.action`: redact,
  warn or block)

### URL Validation

- `validate_url()` rejects non-HTTP(S) schemes and internal, loopback and link-local
  addresses (an SSRF guard), optionally enforces a domain allowlist, then sends a HEAD
  request with a timeout. The scraper passes `url_validation.allowed_domains` and
  `timeout_seconds` from the config
- Used only by `src/tools/scraper.py`, and no agent calls the scraper, so it is not on
  the request path

### Token Budget Enforcement

- Per-run hard cap from `budget.token_budget` (default 50,000), read by `get_token_budget()`
- `check_budget(current_tokens)` returns `False` when budget is exceeded
- Every agent checks budget before making LLM calls
- The check runs between calls, so one call can take a run past the cap

### Prompt Injection Check

- `screen_input(text)` applies the `guardrails.prompt_injection` policy in `configs/base.yaml`
- The planner calls it on the user query before building any prompt
- `action: block` ends the run with an error in `errors`; `action: warn` logs and continues
- The researcher passes every fetched snippet through `screen_sources()`, which calls
  `check_all_guardrails()` and applies the same policy: `block` drops the flagged source,
  `warn` logs it

### Rate Limiting

- `RateLimiter` is a sliding 60-second window (`guardrails.rate_limit.max_rpm`, default 30)
- `throttle()` waits on it before every LLM call and every search-tool call
- Skipped in offline mode and when `guardrails.rate_limit.enabled` is false

## Model and Offline Mode

`get_llm()` in `src/llm.py` is the only place that creates a model client. With a real
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`) it returns `ChatGoogleGenerativeAI` for the model
in `configs/base.yaml` (`gemini-3.6-flash`). Without one, or with `RESEARCH_OFFLINE=1`,
it returns `OfflineLLM`, a rule engine that answers the five agents' prompts with
schema-valid replies and raises on any prompt it does not recognise.

## Graph Construction

```python
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", route_to_researchers, ["researcher"])   # Send() fan-out
graph.add_edge("researcher", "quality_gate")
graph.add_conditional_edges("quality_gate", route_after_quality,
                            {"analyst": "analyst", "retry_researcher": "retry_researcher"})
graph.add_edge("retry_researcher", "analyst")
graph.add_edge("analyst", "synthesizer")
graph.add_edge("synthesizer", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", route_after_review, {"writer": "writer", END: END})
```

A run makes 5 LLM calls with no revision and 7 with one.

## Not on the Request Path

- `src/tools/scraper.py` (`scrape_url`): implemented and tested, never called by an agent.
- `src/cache/research_cache.py`: an SQLite source and query cache; no agent reads or
  writes it, and the Streamlit sidebar only shows its stats.

## Evaluation Architecture

The evaluation framework (`evaluation/`) is kept as a separate top-level module
because it operates independently of the main pipeline:

- **`run_eval.py`:** Runs a single-call baseline against a 3-call prompt chain
  (research, analyse, write) across a test set of 10 research questions (5 by default).
  The chain is not the eight-node graph. Uses LLM-as-judge scoring on accuracy,
  completeness, and citation quality (0-3 scale).
- **`judge_prompt.py`:** The scoring prompts used by the LLM judge. `run_eval.py`
  imports `COMBINED_PROMPT` from here. It used to inline its own copy, so the two
  drifted and editing the rubric in this file changed nothing about the scores.

The evaluation module calls the Gemini SDK (`google.genai`) directly rather than
going through LangGraph, so the single-agent baseline is a genuine one call and not
a framework wrapped around one call.

Two things to know before reading its output:

- **Token caps must leave room for reasoning.** Measured on gemini-3.6-flash, the
  baseline prompt spends ~940 tokens thinking before emitting a word. The original
  1000-2000 caps therefore truncated the reports being scored, and the harness was
  measuring its own configuration rather than the architecture. Truncation is now
  a hard error.
- **Cost is not priced by default.** Tokens are the measurement; dollars are a
  conversion you supply via `PRICE_PER_1M_INPUT` / `PRICE_PER_1M_OUTPUT`. The old
  `PRICE_PER_TOKEN = 0` produced "$0.0000" for both arms and a "Cost multiplier:
  0.0x" for a pipeline that makes three calls against a baseline of one.
