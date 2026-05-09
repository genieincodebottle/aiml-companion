"""Streamlit UI for the Multi-Agent Research Team.

Run: streamlit run ui/app.py
"""

import sys
import os
import time

import streamlit as st

# Add project root to path so we can import agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import run as run_orchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Team",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AI Research Team")
    st.markdown("**Multi-agent system powered by AWS Bedrock AgentCore**")
    st.divider()

    st.markdown("### How It Works")
    st.markdown("""
    1. You enter a research question
    2. **Orchestrator** creates a plan
    3. **Research Agent** searches the web
    4. **Analyst Agent** analyzes findings
    5. **Writer Agent** produces the report
    """)

    st.divider()

    st.markdown("### Example Queries")
    examples = [
        "Compare AI agent frameworks: LangGraph vs CrewAI vs Strands",
        "What are the latest trends in AI-powered code generation?",
        "Analyze the competitive landscape of cloud AI services (AWS vs Azure vs GCP)",
        "Research best practices for deploying LLMs in production",
        "Compare vector databases for RAG: Pinecone vs Weaviate vs pgvector",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["query_input"] = ex
            st.rerun()

    st.divider()
    st.markdown(
        "Built by [Rajesh Srivastava](https://github.com/genieincodebottle) | "
        "[AI/ML Companion](https://aimlcompanion.ai/)"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("Multi-Agent AI Research Team")
st.markdown("*Enter any topic - 3 AI agents will research, analyze, and write a comprehensive report.*")

# Query input
query = st.text_area(
    "What would you like to research?",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="E.g., Compare the top AI agent frameworks for building production applications in 2026",
    key="query_area",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("Research", type="primary", use_container_width=True)
with col2:
    show_steps = st.checkbox("Show step-by-step details", value=False)

# ---------------------------------------------------------------------------
# Execute pipeline
# ---------------------------------------------------------------------------

if run_button and query.strip():
    # Progress tracking
    progress_bar = st.progress(0, text="Creating execution plan...")

    with st.spinner("Agents are working..."):
        start_time = time.time()

        try:
            result = run_orchestrator(query.strip())
            elapsed = time.time() - start_time

            progress_bar.progress(100, text="Complete!")

            # Show step-by-step details
            if show_steps:
                st.divider()
                st.subheader("Execution Details")

                # Plan
                with st.expander("Execution Plan", expanded=False):
                    for step in result["plan"]:
                        agent_emoji = {"research": "🔍", "analyst": "📊", "writer": "✍️"}.get(step["agent"], "🤖")
                        st.markdown(f"{agent_emoji} **{step['agent'].title()}**: {step['task']}")

                # Step results
                for step in result["steps"]:
                    agent_emoji = {"research": "🔍", "analyst": "📊", "writer": "✍️"}.get(step["agent"], "🤖")
                    with st.expander(f"Step {step['step']}: {agent_emoji} {step['agent'].title()} Agent", expanded=False):
                        st.markdown(step["result"])

            # Final report
            st.divider()
            st.subheader("Final Report")
            st.markdown(result["final_report"])

            # Metadata
            st.divider()
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Time", f"{result['total_time']}s")
            with col_b:
                st.metric("Steps Executed", len(result["steps"]))
            with col_c:
                st.metric("Agents Used", len(set(s["agent"] for s in result["steps"])))

        except Exception as e:
            progress_bar.empty()
            st.error(f"Error: {e}")
            st.markdown(
                "**Troubleshooting:**\n"
                "- Check that AWS credentials are configured (`aws configure`)\n"
                "- Verify Bedrock model access is enabled in your region\n"
                "- Check the terminal for detailed error messages"
            )

elif run_button:
    st.warning("Please enter a research query.")
