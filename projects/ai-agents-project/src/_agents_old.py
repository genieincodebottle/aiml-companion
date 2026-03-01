# ============================================
# Multi-Agent Research System with LangGraph
# ============================================
# uv pip install langgraph langchain-google-genai langchain-tavily

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from src.guardrails import scrub_pii, validate_url, check_budget, TOKEN_BUDGET

load_dotenv()


# === 1. Define Shared State ===
class ResearchState(TypedDict):
    query: str                    # User's research question
    sources: list[dict]           # Retrieved sources
    key_claims: list[dict]        # Extracted claims with citations
    draft: str                    # Written report draft
    fact_check: list[dict]        # Verification results
    final_report: str             # Polished output
    token_count: int              # Cumulative token tracking
    errors: list[str]             # Error log


# === 2. Specialized Agent Nodes ===
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
search_tool = TavilySearch(max_results=5)


def researcher(state: ResearchState) -> dict:
    """Search for relevant sources on the topic."""
    if not check_budget(state.get("token_count", 0)):
        return {"errors": state.get("errors", []) + ["Budget exceeded before research"]}
    try:
        raw = search_tool.invoke(state["query"])
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        sources = [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("content", "")[:500], "date": r.get("date", "")}
            for r in results
        ]
        return {"sources": sources, "token_count": state.get("token_count", 0) + 500}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Researcher error: {e}"],
                "sources": [], "token_count": state.get("token_count", 0)}


def analyst(state: ResearchState) -> dict:
    """Extract structured claims from sources."""
    if not check_budget(state.get("token_count", 0)):
        return {"errors": state.get("errors", []) + ["Budget exceeded before analysis"]}
    if not state.get("sources"):
        return {"key_claims": [], "errors": state.get("errors", []) + ["No sources to analyze"]}
    try:
        source_text = "\n".join(
            f"[{i+1}] {s['title']}: {s['snippet']}" for i, s in enumerate(state["sources"])
        )
        response = llm.invoke(
            f"Extract 5-8 key claims from these sources. "
            f"Format each as: claim | source_number | confidence (high/medium/low)\n\n"
            + source_text
        )
        claims = []
        for line in response.content.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 2:
                claims.append({
                    "claim": parts[0].strip(),
                    "source_idx": parts[1].strip(),
                    "confidence": parts[2].strip() if len(parts) > 2 else "medium"
                })
        tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 1000)
        return {"key_claims": claims, "token_count": state.get("token_count", 0) + tokens}
    except Exception as e:
        return {"key_claims": [], "errors": state.get("errors", []) + [f"Analyst error: {e}"],
                "token_count": state.get("token_count", 0)}


def writer(state: ResearchState) -> dict:
    """Write structured report from claims (ONLY using provided sources)."""
    if not check_budget(state.get("token_count", 0)):
        return {"errors": state.get("errors", []) + ["Budget exceeded before writing"]}
    try:
        claims_text = "\n".join(
            f"- {c['claim']} [Source {c['source_idx']}]" for c in state["key_claims"]
        )
        sources_text = "\n".join(
            f"[{i+1}] {s['title']} - {s['url']}" for i, s in enumerate(state["sources"])
        )
        response = llm.invoke(
            f"Write a structured research report using ONLY these claims and sources.\n\n"
            f"Claims:\n{claims_text}\n\nSources:\n{sources_text}\n\n"
            f"Format: ## Introduction, ## Key Findings, ## Analysis, ## Sources"
        )
        tokens = response.response_metadata.get("token_usage", {}).get("total_tokens", 1500)
        return {"draft": response.content, "token_count": state.get("token_count", 0) + tokens}
    except Exception as e:
        return {"draft": "", "errors": state.get("errors", []) + [f"Writer error: {e}"],
                "token_count": state.get("token_count", 0)}


def fact_checker(state: ResearchState) -> dict:
    """Validate claims against sources, then revise the report."""
    if not check_budget(state.get("token_count", 0)):
        return {"final_report": state.get("draft", ""),
                "errors": state.get("errors", []) + ["Budget exceeded before fact-check"]}
    try:
        source_summaries = "\n".join(
            f"[{i+1}] {s['snippet'][:200]}" for i, s in enumerate(state["sources"])
        )
        # Step 1: Identify unsupported claims
        check_response = llm.invoke(
            f"Verify each claim in this report against the source list. "
            f"For each unsupported claim, output exactly: UNSUPPORTED: <the claim text>\n\n"
            f"Report:\n{state['draft'][:2000]}\n\nSources:\n" + source_summaries
        )
        check_tokens = check_response.response_metadata.get(
            "token_usage", {}).get("total_tokens", 1000)

        # Step 2: Revise the report - replace unsupported claims with markers
        revised = state["draft"]
        unsupported = []
        for line in check_response.content.split("\n"):
            if line.strip().startswith("UNSUPPORTED:"):
                claim = line.replace("UNSUPPORTED:", "").strip()
                unsupported.append(claim)
                if claim and claim in revised:
                    revised = revised.replace(claim, "[NEEDS CITATION] " + claim)

        # Step 3: Scrub PII from final output
        revised, pii_found = scrub_pii(revised)
        pii_note = f" PII redacted: {pii_found}" if pii_found else ""

        # Step 4: Validate source URLs
        valid_urls = 0
        for src in state.get("sources", []):
            if validate_url(src.get("url", "")):
                valid_urls += 1

        total_tokens = state.get("token_count", 0) + check_tokens
        return {
            "fact_check": [{
                "unsupported_claims": unsupported,
                "pii_redacted": pii_found,
                "valid_urls": f"{valid_urls}/{len(state.get('sources', []))}",
            }],
            "final_report": revised,
            "token_count": total_tokens,
            "errors": state.get("errors", []) + ([pii_note] if pii_note else [])
        }
    except Exception as e:
        return {"fact_check": [{"error": str(e)}],
                "final_report": state.get("draft", ""),
                "errors": state.get("errors", []) + [f"Fact-checker error: {e}"],
                "token_count": state.get("token_count", 0)}


# === 3. Build the Graph ===
def build_graph():
    """Build and compile the research pipeline graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)
    graph.add_node("fact_checker", fact_checker)

    # Sequential pipeline: research -> analyze -> write -> verify + revise
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "fact_checker")
    graph.add_edge("fact_checker", END)

    return graph.compile()


def run_pipeline(query: str) -> dict:
    """Run the full multi-agent research pipeline."""
    app = build_graph()
    result = app.invoke({
        "query": query,
        "sources": [], "key_claims": [], "draft": "",
        "fact_check": [], "final_report": "",
        "token_count": 0, "errors": []
    })
    return result


if __name__ == "__main__":
    result = run_pipeline("What are the latest trends in AI agents for 2025?")
    print(f"Report length: {len(result['final_report'])} chars")
    print(f"Sources found: {len(result['sources'])}")
    print(f"Claims extracted: {len(result['key_claims'])}")
    print(f"Total tokens used: {result['token_count']} / {TOKEN_BUDGET}")
    print(f"Fact-check results: {result['fact_check']}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")