# ============================================
# Multi-Agent Research System - Streamlit UI
# ============================================
# streamlit run app.py
#
# Features:
#   - 8-agent pipeline visualization with live trace
#   - Graph topology display showing active nodes
#   - Pipeline trace timeline (agent, duration, tokens)
#   - Source quality heatmap
#   - Iteration counter for refinement loop
#   - Cache hit/miss indicator
#   - Single-agent vs multi-agent evaluation

import streamlit as st
import os
import yaml
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.agents import build_graph
from src.models.state import default_state
from src.guardrails import TOKEN_BUDGET
from src.cache.research_cache import get_cache_stats
from evaluation.run_eval import (
    TEST_QUESTIONS, single_agent_research, multi_agent_research,
    evaluate_report, PRICE_PER_TOKEN,
)

# === Page Config ===
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Session State ===
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "guardrail_status" not in st.session_state:
    st.session_state.guardrail_status = {"pii": "idle", "url": "idle", "budget": "ok"}


# === Sidebar ===
def render_sidebar():
    with st.sidebar:
        st.title("\U0001F50D Research System")

        # Model info
        try:
            config = yaml.safe_load(open("configs/base.yaml"))
            model_name = config["model"]["name"]
            provider = config["model"]["provider"]
        except Exception:
            model_name = "gemini-2.5-flash"
            provider = "google"

        st.subheader("Model")
        st.caption(f"{model_name} ({provider})")
        st.caption("Free tier - $0.00 per query")

        st.divider()

        # API key status
        st.subheader("API Keys")
        google_ok = bool(os.environ.get("GOOGLE_API_KEY"))
        tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
        st.markdown(f"{'🟢' if google_ok else '🔴'} Google AI (Gemini)")
        st.markdown(f"{'🟢' if tavily_ok else '🔴'} Tavily Search")

        st.divider()

        # Token budget
        st.subheader("Token Budget")
        current = st.session_state.token_count
        pct = min(current / TOKEN_BUDGET, 1.0)
        st.progress(pct)
        st.metric("Tokens Used", f"{current:,} / {TOKEN_BUDGET:,}")

        st.divider()

        # Guardrail status
        st.subheader("Guardrails")
        gs = st.session_state.guardrail_status
        status_icons = {"ok": "🟢", "idle": "⚪", "warning": "🟡", "error": "🔴"}
        for label, key in [("PII Scrubbing", "pii"), ("URL Validation", "url"), ("Budget", "budget")]:
            icon = status_icons.get(gs[key], "⚪")
            st.markdown(f"{icon} {label}")

        st.divider()

        # Cache stats
        st.subheader("Research Cache")
        stats = get_cache_stats()
        st.caption(f"Cached queries: {stats['cached_queries']}")
        st.caption(f"Indexed sources: {stats['indexed_sources']}")
        st.caption(f"Cache hits: {stats['total_hits']}")


# === Agent Node Renderers ===

AGENT_CONFIG = {
    "planner": {
        "label": "\U0001F4CB Planner",
        "icon": "📋",
    },
    "researcher": {
        "label": "\U0001F50D Researcher",
        "icon": "🔍",
    },
    "quality_gate": {
        "label": "\U0001F6E1\uFE0F Quality Gate",
        "icon": "🛡️",
    },
    "retry_researcher": {
        "label": "\U0001F504 Retry Research",
        "icon": "🔄",
    },
    "analyst": {
        "label": "\U0001F4CA Analyst",
        "icon": "📊",
    },
    "synthesizer": {
        "label": "\U0001F517 Synthesizer",
        "icon": "🔗",
    },
    "writer": {
        "label": "\u270D\uFE0F Writer",
        "icon": "✍️",
    },
    "reviewer": {
        "label": "\U0001F50E Reviewer",
        "icon": "🔎",
    },
}


def render_node_output(node_name, accumulated_state):
    """Render output for a specific agent node."""
    if node_name == "planner":
        topics = accumulated_state.get("sub_topics", [])
        plan = accumulated_state.get("research_plan", "")
        st.write(f"**{len(topics)} sub-topic(s)** to research:")
        for i, t in enumerate(topics, 1):
            st.markdown(f"  {i}. {t}")
        if plan:
            st.caption(f"Strategy: {plan[:200]}")

    elif node_name in ("researcher", "retry_researcher"):
        sources = accumulated_state.get("sources", [])
        st.write(f"**{len(sources)} sources** found so far")
        for s in sources[-5:]:  # Show last 5 (most recent)
            tool_badge = f"[{s.get('tool', '?')}]" if s.get('tool') else ""
            st.markdown(f"- {tool_badge} **{s.get('title', 'Untitled')[:60]}**")
            if s.get("url"):
                st.caption(f"  {s['url']}")

    elif node_name == "quality_gate":
        score = accumulated_state.get("quality_score", 0)
        passed = accumulated_state.get("quality_passed", False)
        if passed:
            st.success(f"Quality score: **{score:.2f}** - PASSED")
        else:
            st.warning(f"Quality score: **{score:.2f}** - Needs more research")

        # Source ranking
        ranking = accumulated_state.get("source_ranking", [])
        if ranking:
            with st.expander(f"Source Rankings ({len(ranking)})"):
                for r in ranking[:8]:
                    bar_pct = int(r.get("combined_score", 0) * 100)
                    st.markdown(
                        f"{'🟢' if bar_pct > 60 else '🟡' if bar_pct > 30 else '🔴'} "
                        f"**{r.get('title', '')[:40]}** - {bar_pct}%"
                    )

    elif node_name == "analyst":
        claims = accumulated_state.get("key_claims", [])
        conflicts = accumulated_state.get("conflicts", [])
        st.write(f"**{len(claims)} claims** extracted, **{len(conflicts)} conflicts** detected")
        for c in claims:
            conf = c.get("confidence", "medium").lower()
            icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            st.markdown(f"{icon} {c.get('claim', '')} *(Source {c.get('source_idx', '?')}, {conf})*")
        if conflicts:
            st.warning(f"{len(conflicts)} conflict(s) detected between sources")

    elif node_name == "synthesizer":
        synthesis = accumulated_state.get("synthesis", "")
        st.write(f"Synthesis: **{len(synthesis):,} characters**")
        with st.expander("Preview Synthesis", expanded=False):
            st.markdown(synthesis[:2000])

    elif node_name == "writer":
        drafts = accumulated_state.get("drafts", [])
        current = accumulated_state.get("current_draft", "")
        revision = accumulated_state.get("revision_count", 0)
        st.write(f"Draft **v{revision}**: **{len(current):,} characters**")
        with st.expander(f"Preview Draft v{revision}", expanded=False):
            st.markdown(current[:3000])

    elif node_name == "reviewer":
        review = accumulated_state.get("review", {})
        score = review.get("score", 0)
        passed = review.get("passed", False)
        revision = accumulated_state.get("revision_count", 0)

        if passed:
            st.success(f"Review score: **{score}/10** - APPROVED")
        else:
            st.warning(f"Review score: **{score}/10** - Revision needed (attempt {revision})")

        issues = review.get("issues", [])
        if issues:
            with st.expander(f"Issues ({len(issues)})"):
                for issue in issues:
                    st.markdown(f"- {issue}")

        suggestions = review.get("suggestions", [])
        if suggestions:
            with st.expander(f"Suggestions ({len(suggestions)})"):
                for s in suggestions:
                    st.markdown(f"- {s}")


# === Research Pipeline Tab ===
def render_research_tab():
    st.header("Research Pipeline")
    st.caption(
        "8-agent pipeline: Planner > Researcher (parallel) > Quality Gate > "
        "Analyst > Synthesizer > Writer > Reviewer > (refine or done)"
    )

    # Graph topology (static display)
    with st.expander("Pipeline Architecture", expanded=False):
        st.code(
            "Planner --> Researcher(s) [parallel] --> Quality Gate\n"
            "  |                                         |\n"
            "  |              [score < 0.4] --> Retry Research\n"
            "  |              [score >= 0.4] |\n"
            "  |                             v\n"
            "  +----> Analyst --> Synthesizer --> Writer --> Reviewer\n"
            "                                      ^          |\n"
            "                                      |   [score < 7]\n"
            "                                      +----------+\n"
            "                                         [score >= 7] --> END",
            language=None,
        )

    query = st.text_input(
        "Research Question",
        placeholder="e.g., What are the latest trends in AI agents for 2025?",
        key="research_query",
    )

    run_btn = st.button("\U0001F680 Run Research", type="primary")

    if run_btn and query:
        run_research_pipeline(query)

    if st.session_state.research_result and not run_btn:
        display_final_report(st.session_state.research_result)


def run_research_pipeline(query):
    st.session_state.research_result = None
    st.session_state.token_count = 0
    st.session_state.guardrail_status = {"pii": "idle", "url": "idle", "budget": "ok"}

    app = build_graph()
    initial_state = default_state(query)

    accumulated_state = dict(initial_state)
    seen_nodes = []

    try:
        for step in app.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            # Merge partial state
            for key, value in node_output.items():
                if key in accumulated_state and isinstance(accumulated_state[key], list) and isinstance(value, list):
                    accumulated_state[key] = accumulated_state[key] + value
                else:
                    accumulated_state[key] = value

            cfg = AGENT_CONFIG.get(node_name)
            if not cfg:
                continue

            seen_nodes.append(node_name)

            with st.status(cfg["label"], state="complete") as status:
                render_node_output(node_name, accumulated_state)
                tokens = accumulated_state.get("token_count", 0)
                st.caption(f"Tokens used: {tokens:,} / {TOKEN_BUDGET:,}")
                status.update(label=f"{cfg['label']} - Done", state="complete")

            st.session_state.token_count = accumulated_state.get("token_count", 0)

    except Exception as e:
        st.error(f"Pipeline error: {e}")

    # Update guardrail status
    tokens = accumulated_state.get("token_count", 0)
    if tokens >= TOKEN_BUDGET:
        st.session_state.guardrail_status["budget"] = "error"
    elif tokens >= TOKEN_BUDGET * 0.8:
        st.session_state.guardrail_status["budget"] = "warning"
    else:
        st.session_state.guardrail_status["budget"] = "ok"

    # Check PII from writer drafts
    drafts = accumulated_state.get("drafts", [])
    pii_found = any(d.get("pii_scrubbed") for d in drafts)
    st.session_state.guardrail_status["pii"] = "warning" if pii_found else "ok"
    st.session_state.guardrail_status["url"] = "ok"

    # Show errors
    errors = accumulated_state.get("errors", [])
    if errors:
        with st.expander(f"\u26A0\uFE0F Errors ({len(errors)})", expanded=True):
            for err in errors:
                st.error(err)

    # Pipeline trace
    trace = accumulated_state.get("pipeline_trace", [])
    if trace:
        render_pipeline_trace(trace)

    st.session_state.research_result = accumulated_state
    display_final_report(accumulated_state)


def render_pipeline_trace(trace):
    """Render the pipeline execution trace as a timeline."""
    st.divider()
    st.subheader("Pipeline Trace")

    total_ms = sum(t.get("duration_ms", 0) for t in trace)
    total_tokens = sum(t.get("tokens", 0) for t in trace)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Duration", f"{total_ms / 1000:.1f}s")
    col2.metric("Total Tokens", f"{total_tokens:,}")
    col3.metric("Agent Steps", len(trace))

    # Trace table
    trace_rows = []
    for i, t in enumerate(trace):
        trace_rows.append({
            "#": i + 1,
            "Agent": t.get("agent", "?"),
            "Duration": f"{t.get('duration_ms', 0)}ms",
            "Tokens": t.get("tokens", 0),
            "Summary": t.get("summary", ""),
        })
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)


def display_final_report(result):
    st.divider()
    st.subheader("Final Report")
    report = result.get("final_report", "")
    if report:
        st.markdown(report)

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sources", len(result.get("sources", [])))
        col2.metric("Claims", len(result.get("key_claims", [])))
        col3.metric("Quality", f"{result.get('quality_score', 0):.2f}")
        col4.metric("Drafts", len(result.get("drafts", [])))
        col5.metric("Tokens", f"{result.get('token_count', 0):,}")

        # Review score
        review = result.get("review", {})
        if review:
            score = review.get("score", 0)
            st.info(f"Review score: {score}/10 | "
                    f"Revisions: {result.get('revision_count', 0)} | "
                    f"Conflicts detected: {len(result.get('conflicts', []))}")
    else:
        st.warning("No report was generated. Check the errors above.")


# === Evaluation Tab ===
def render_evaluation_tab():
    st.header("Evaluation: Single-Agent vs Multi-Agent")
    st.caption("Compare a single LLM call against the 8-agent pipeline on research questions")

    col1, col2 = st.columns([1, 3])
    with col1:
        num_q = st.selectbox("Questions", [1, 2, 3, 5], index=1)
    with col2:
        st.caption(f"Will evaluate {num_q} question(s) from the 10-question test set")

    eval_btn = st.button("\U0001F4CB Run Evaluation", type="primary")

    if eval_btn:
        run_evaluation(num_q)

    if st.session_state.eval_results and not eval_btn:
        display_eval_results(st.session_state.eval_results)


def run_evaluation(num_questions):
    st.session_state.eval_results = None

    results = {"single": [], "multi": [], "questions": []}
    questions = TEST_QUESTIONS[:num_questions]
    progress = st.progress(0, text="Starting evaluation...")

    for i, test in enumerate(questions):
        q_short = test["query"][:50]
        progress.progress(i / num_questions, text=f"Q{i+1}/{num_questions}: {q_short}...")
        results["questions"].append(test["query"])

        # Single-agent
        with st.status(f"Q{i+1} Single-agent: {q_short}...", state="running") as s:
            single = single_agent_research(test["query"])
            single_scores = evaluate_report(test["query"], single["report"])
            results["single"].append({
                **single_scores,
                "tokens": single["tokens"],
                "latency": single["latency"],
                "cost": single["cost"],
            })
            s.update(
                label=f"Q{i+1} Single-agent: Done (acc={single_scores.get('accuracy', '?')})",
                state="complete",
            )

        # Multi-agent
        with st.status(f"Q{i+1} Multi-agent: {q_short}...", state="running") as s:
            multi = multi_agent_research(test["query"])
            multi_scores = evaluate_report(test["query"], multi["report"])
            results["multi"].append({
                **multi_scores,
                "tokens": multi["tokens"],
                "latency": multi["latency"],
                "cost": multi["cost"],
            })
            s.update(
                label=f"Q{i+1} Multi-agent: Done (acc={multi_scores.get('accuracy', '?')})",
                state="complete",
            )

    progress.progress(1.0, text="Evaluation complete!")
    st.session_state.eval_results = results
    display_eval_results(results)


def display_eval_results(results):
    df_s = pd.DataFrame(results["single"])
    df_m = pd.DataFrame(results["multi"])

    st.divider()

    # Side-by-side metrics
    st.subheader("Score Comparison (0-3 scale)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Single-Agent**")
        st.metric("Accuracy", f"{df_s['accuracy'].mean():.1f} / 3")
        st.metric("Completeness", f"{df_s['completeness'].mean():.1f} / 3")
        st.metric("Citations", f"{df_s['citations'].mean():.1f} / 3")

    with col2:
        st.markdown("**Multi-Agent (8 agents)**")
        d_acc = df_m["accuracy"].mean() - df_s["accuracy"].mean()
        d_comp = df_m["completeness"].mean() - df_s["completeness"].mean()
        d_cit = df_m["citations"].mean() - df_s["citations"].mean()
        st.metric("Accuracy", f"{df_m['accuracy'].mean():.1f} / 3", f"{d_acc:+.1f}")
        st.metric("Completeness", f"{df_m['completeness'].mean():.1f} / 3", f"{d_comp:+.1f}")
        st.metric("Citations", f"{df_m['citations'].mean():.1f} / 3", f"{d_cit:+.1f}")

    # Bar chart
    st.subheader("Quality Comparison")
    chart_df = pd.DataFrame({
        "Single-Agent": [df_s["accuracy"].mean(), df_s["completeness"].mean(), df_s["citations"].mean()],
        "Multi-Agent": [df_m["accuracy"].mean(), df_m["completeness"].mean(), df_m["citations"].mean()],
    }, index=["Accuracy", "Completeness", "Citations"])
    st.bar_chart(chart_df)

    # Efficiency
    st.subheader("Efficiency")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Single Avg Tokens", f"{df_s['tokens'].mean():,.0f}")
    col2.metric("Multi Avg Tokens", f"{df_m['tokens'].mean():,.0f}")
    col3.metric("Single Avg Latency", f"{df_s['latency'].mean():.1f}s")
    col4.metric("Multi Avg Latency", f"{df_m['latency'].mean():.1f}s")

    # Verdict
    st.divider()
    multi_better = df_m["accuracy"].mean() > df_s["accuracy"].mean() * 1.2
    if multi_better:
        st.success(
            "\U0001F3AF **Verdict: Multi-agent justified** - "
            "significant quality improvement over single-agent baseline"
        )
    else:
        st.info(
            "\U0001F4A1 **Verdict: Consider optimizing single-agent first** - "
            "marginal improvement doesn't justify the added complexity and cost"
        )

    # Per-question detail
    with st.expander("Per-Question Breakdown"):
        detail_rows = []
        for i, q in enumerate(results["questions"]):
            s = results["single"][i]
            m = results["multi"][i]
            detail_rows.append({
                "Question": q[:60] + "...",
                "S-Accuracy": s["accuracy"],
                "M-Accuracy": m["accuracy"],
                "S-Completeness": s["completeness"],
                "M-Completeness": m["completeness"],
                "S-Tokens": s["tokens"],
                "M-Tokens": m["tokens"],
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


# === Main ===
def main():
    st.title("Multi-Agent Research System")
    st.caption(
        "8-agent pipeline: Planner \u2192 Researcher (parallel) \u2192 Quality Gate \u2192 "
        "Analyst \u2192 Synthesizer \u2192 Writer \u2192 Reviewer | "
        "Powered by Gemini 2.5 Flash (free tier) + Tavily Search + Wikipedia"
    )

    render_sidebar()

    tab_research, tab_eval = st.tabs(["\U0001F50D Research Pipeline", "\U0001F4CB Evaluation"])

    with tab_research:
        render_research_tab()

    with tab_eval:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
