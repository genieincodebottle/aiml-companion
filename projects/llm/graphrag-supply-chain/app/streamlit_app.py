"""GraphRAG Supply Chain Intelligence - the learning UI.

A THIN CLIENT. Every line of business logic - retrieval, Cypher, prompts,
guardrails, scoring - lives behind the FastAPI backend in `api/`. This file
calls HTTP endpoints and renders what comes back. It contains no domain rules
and makes no decisions the backend has not already made.

The design rule throughout: never assert, always show. Where the UI says a
traversal happened, it prints the Cypher that ran and the rows that came back.
Where it says GraphRAG won, it shows the measurement, including the questions
where it loses.

Run it with the backend up:

    python run.py api      # terminal 1
    python run.py app      # terminal 2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import ApiClient, ApiError  # noqa: E402

st.set_page_config(
    page_title="GraphRAG Supply Chain",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling. Restrained on purpose: this is a tool for reading evidence, and
# decoration that competes with the content makes it harder to read.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
      h1, h2, h3 { letter-spacing: -0.015em; }
      [data-testid="stMetricValue"] { font-size: 1.45rem; }
      [data-testid="stMetricLabel"] { font-size: 0.78rem; opacity: 0.75; }
      .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
      .stTabs [data-baseweb="tab"] {
          padding: 0.55rem 1.05rem; border-radius: 7px 7px 0 0; font-weight: 500;
      }
      code { font-size: 0.86em; }
      .pill {
          display: inline-block; padding: 0.16rem 0.6rem; border-radius: 999px;
          font-size: 0.74rem; font-weight: 600; margin-right: 0.35rem;
          border: 1px solid currentColor;
      }
      .pill-pass  { color: #1a7f37; background: #1a7f3714; }
      .pill-warn  { color: #9a6700; background: #9a670014; }
      .pill-block { color: #b42318; background: #b4231814; }
      .pill-erp   { color: #0b62d1; background: #0b62d114; }
      .pill-llm   { color: #7b3fbf; background: #7b3fbf14; }
      .evidence {
          border-left: 3px solid #8884; padding: 0.35rem 0 0.35rem 0.8rem;
          margin: 0.35rem 0 0.9rem 0;
      }
      .muted { opacity: 0.7; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

TYPE_COLOURS = {
    "Product": "#1f77b4", "Component": "#ff7f0e", "Supplier": "#2ca02c",
    "Site": "#9467bd", "Location": "#8c564b", "Material": "#e377c2",
    "Incident": "#d62728", "Finding": "#bcbd22", "Regulation": "#17becf",
    "Certification": "#7f7f7f",
}

SAMPLE_QUESTIONS = [
    "Typhoon Meilin has shut down Kaohsiung for about three weeks. Which of our finished products are exposed, and through which suppliers?",
    "Is our dual sourcing on the LI-18650 battery pack genuine, or do the two suppliers share an upstream dependency?",
    "If Sarawak Copper Foil stopped shipping, which Northwind products would eventually be affected?",
    "Which sole-sourced components come from a supplier that currently has an open audit finding?",
    "Which of our products fall under the EU Battery Regulation due diligence obligations?",
    "What did the 2026 Prahara Polymers audit say about the fire suppression system?",
    "Under our dual sourcing policy, what makes a component require a second source?",
    "Who manufactures the TFT panel inside the DSP-3300 display module?",
]


@st.cache_resource(show_spinner=False)
def get_client() -> ApiClient:
    return ApiClient()


def pill(text: str, kind: str = "pass") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def show_api_error(exc: ApiError) -> None:
    if exc.kind == "connection":
        st.error("The backend is not running")
        st.code(str(exc), language="text")
    elif exc.kind in {"prompt_injection", "rate_limit", "input_length",
                      "secret_in_document"}:
        st.warning(f"Blocked by a guardrail: {exc.kind}")
        st.write(str(exc))
        if exc.detail:
            with st.expander("What the scanner matched"):
                st.json(exc.detail)
    else:
        st.error(str(exc))


def render_graph(nodes: list[dict], edges: list[dict],
                 seeds: set[str] | None = None) -> None:
    """Draw a subgraph with Graphviz.

    DOT rather than a JavaScript graph widget: Streamlit renders it client-side
    with no extra dependency, the layout is deterministic so the same subgraph
    looks the same every time, and the DOT source is itself readable - which
    matters in a teaching tool.
    """
    if not nodes:
        st.info("Nothing to draw for this selection.")
        return

    seeds = seeds or set()
    lines = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; splines=true; overlap=false;',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];',
        '  edge [fontname="Helvetica" fontsize=8 color="#77777799"];',
    ]
    for node in nodes:
        colour = TYPE_COLOURS.get(node.get("type", ""), "#999999")
        label = str(node.get("name", "")).replace('"', "'")
        if len(label) > 32:
            label = label[:29] + "..."
        is_seed = node["key"] in seeds or node.get("is_seed")
        lines.append(
            f'  "{node["key"]}" [label="{label}\\n({node.get("type", "")})" '
            f'fillcolor="{colour}22" color="{colour}" '
            f'penwidth={3 if is_seed else 1}];'
        )
    for edge in edges:
        # Dashed = extracted from prose by the model. Solid = system of record.
        # A user must never have to guess which edges are inferences.
        style = "dashed" if edge.get("provenance") == "llm" else "solid"
        lines.append(
            f'  "{edge["source"]}" -> "{edge["target"]}" '
            f'[label="{edge["type"]}" style={style}];'
        )
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)
    st.markdown(
        '<span class="muted">'
        + pill("solid = ERP / PLM", "erp")
        + pill("dashed = LLM-extracted", "llm")
        + " Extracted edges carry a confidence score and the verbatim sentence "
          "they came from.</span>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Sidebar
# ===========================================================================
def sidebar(client: ApiClient) -> dict:
    st.sidebar.title("GraphRAG Supply Chain")
    st.sidebar.caption("Multi-tier supply chain risk intelligence on Neo4j.")

    try:
        health = client.health()
    except ApiError as exc:
        st.sidebar.error("Backend unreachable")
        st.sidebar.code(str(exc)[:300], language="text")
        return {}

    if not health.get("graph_populated"):
        st.sidebar.warning("The graph is empty. Build it on the **Build** tab.")
    else:
        try:
            census = client.census()
            col1, col2 = st.sidebar.columns(2)
            col1.metric("Nodes", census["total_nodes"])
            col2.metric("Edges", census["total_relationships"])
            st.sidebar.metric(
                "Sub-tier dependencies", census["relationships"].get("DEPENDS_ON", 0),
                help="DEPENDS_ON edges. Every one was extracted from prose - "
                     "none of them exist in the ERP, which is the entire reason "
                     "this project exists.",
            )
        except ApiError:
            pass

    st.sidebar.divider()
    st.sidebar.markdown("**Status**")
    st.sidebar.markdown(
        pill("guardrails on" if health.get("guardrails_enabled") else "guardrails OFF",
             "pass" if health.get("guardrails_enabled") else "block")
        + pill("authenticated" if health.get("authenticated") else "open API",
               "pass" if health.get("authenticated") else "warn"),
        unsafe_allow_html=True,
    )
    st.sidebar.code(
        f"model      {health.get('model', '?')}\n"
        f"dimensions {health.get('embedding_dimensions', '?')}\n"
        f"api        {client.base_url}",
        language="text",
    )
    st.sidebar.caption(
        "This UI is a thin HTTP client. All retrieval, Cypher, prompting and "
        "guardrail logic runs in the FastAPI backend - open "
        f"{client.base_url}/docs to call it directly."
    )
    return health


# ===========================================================================
# Build
# ===========================================================================
def tab_build(client: ApiClient) -> None:
    st.header("Build the knowledge graph")
    st.markdown(
        "Ingestion turns a folder of documents plus a few CSV exports into a "
        "graph you can traverse. The two sources are treated differently, and "
        "that split is the most important design decision in the project."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "#### Structured source → loaded verbatim\n"
            "`data/structured/*.csv`, standing in for an ERP and PLM export.\n\n"
            "- Products, components, bill of materials\n"
            "- Tier-1 contracts and sole-source flags\n"
            "- Sites and their locations\n\n"
            "**No LLM touches this.** These facts are already structured, "
            "governed and correct. Running them through an extractor would take "
            "data at 100% accuracy and hand it back at 95%, at a cost, for no "
            "benefit."
        )
    with col2:
        st.markdown(
            "#### Unstructured source → LLM extraction\n"
            "`data/documents/*.md`, standing in for audit reports, "
            "questionnaires, incident bulletins and regulatory notices.\n\n"
            "- **Tier-2 and tier-3 dependencies** - the whole point\n"
            "- Incidents, findings, certifications, regulations\n\n"
            "This is the information no ERP has, because the company has no "
            "commercial relationship with these suppliers and never places an "
            "order with them."
        )

    st.divider()
    status = client.ingest_status()

    if status.get("running"):
        st.info("Ingestion is running.")
        bar = st.progress(status.get("progress", 0.0),
                          text=status.get("message", "working"))
        # Poll rather than block. The backend runs the job in a thread, so a
        # two-minute HTTP request would be killed by a proxy long before it
        # returned - and the work would continue with nobody to receive it.
        while status.get("running"):
            time.sleep(2)
            status = client.ingest_status()
            bar.progress(min(status.get("progress", 0.0), 1.0),
                         text=status.get("message", "working"))
        st.rerun()

    reset = st.checkbox("Wipe the graph first", value=True,
                        help="Required after changing the schema or the "
                             "embedding dimension.")
    st.caption(
        "About 33 LLM calls and two minutes. Embeddings are cached on disk, so "
        "a second run costs nothing for the embedding step."
    )
    if st.button("Run ingestion", type="primary"):
        try:
            client.start_ingest(reset=reset)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)

    if status.get("error"):
        st.error("Ingestion failed")
        st.code(status["error"], language="text")
        st.info(
            "The pipeline verifies the graph before reporting success, so a "
            "failure here means the graph would not have been usable. That is "
            "deliberate: an ingestion that reports success and leaves a broken "
            "graph is worse than one that fails, because the damage surfaces "
            "later as bad answers instead of as an error."
        )

    report = status.get("report")
    if report:
        st.success(f"Graph built in {report.get('seconds', 0):.0f}s")
        cols = st.columns(5)
        cols[0].metric("Documents", report.get("documents", 0))
        cols[1].metric("Chunks", report.get("chunks", 0))
        cols[2].metric("Entities", report.get("entities", 0))
        cols[3].metric("Relationships", report.get("relationships", 0))
        cols[4].metric("Mentions", report.get("mentions", 0))

        blocked = report.get("blocked_documents") or []
        if blocked:
            st.error(f"{len(blocked)} document(s) were blocked by the ingest "
                     "guardrail and never reached the extractor")
            st.dataframe(pd.DataFrame(blocked), use_container_width=True,
                         hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("##### Entity resolution")
            st.caption(
                "`new` is a genuinely new entity. `retyped` means the model gave "
                "a known part the wrong type and the ERP's type won. "
                "`rejected_fuzzy` means the resolver saw a close-but-not-close-"
                "enough match and declined to merge - a spike there means the "
                "corpus has naming drift a human should look at."
            )
            st.dataframe(pd.DataFrame([report.get("resolver", {})]),
                         use_container_width=True, hide_index=True)
        with right:
            st.markdown("##### Cost of this run")
            st.dataframe(pd.DataFrame([report.get("usage", {})]),
                         use_container_width=True, hide_index=True)

        with st.expander("Graph census"):
            counts = report.get("graph_counts", {})
            st.dataframe(
                pd.DataFrame([{"label": k, "count": v} for k, v in counts.items()]),
                use_container_width=True, hide_index=True,
            )


# ===========================================================================
# Explore
# ===========================================================================
def tab_explore(client: ApiClient) -> None:
    st.header("Explore the graph")

    census = client.census()
    if not census.get("populated"):
        st.warning("The graph is empty. Build it on the **Build** tab first.")
        return

    types = sorted(census["nodes"])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        entity_type = st.selectbox("Type", ["(any)"] + types)
    with col2:
        search = st.text_input("Search by name", placeholder="e.g. Formosa")
    with col3:
        hops = st.slider("Hops", 1, 3, 2)

    entities = client.entities(
        entity_type=None if entity_type == "(any)" else entity_type,
        search=search or None,
    )
    if not entities:
        st.info("No entities match that filter.")
        return

    options = {f"{e['name']}  ·  {e['type']}": e["key"] for e in entities}
    label = st.selectbox("Entity", list(options))
    key = options[label]

    detail = client.entity(key)
    graph = client.subgraph([key], hops=hops)

    head1, head2 = st.columns([3, 1])
    with head1:
        st.subheader(detail["name"])
        st.caption(f"{detail['type']}  ·  `{detail['key']}`")
        if detail.get("summary"):
            st.write(detail["summary"])
        if detail.get("aliases"):
            st.caption("Also written as: " + ", ".join(detail["aliases"]))
            st.markdown(
                '<span class="muted">These aliases are how the entity linker '
                'finds this node from a question that uses a short name. '
                'Discarding them at merge time is what makes such a question '
                'silently fail.</span>', unsafe_allow_html=True,
            )
    with head2:
        st.metric("Relationships", len(detail.get("neighbours", [])))
        st.metric("Documents", len(detail.get("documents", [])))

    render_graph(graph["nodes"], graph["edges"], seeds={key})

    tabs = st.tabs(["Relationships", "Evidence behind each edge", "Documents"])
    with tabs[0]:
        rows = detail.get("neighbours", [])
        if rows:
            frame = pd.DataFrame(rows)[["direction", "rel", "name", "type"]]
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("This entity has no relationships.")

    with tabs[1]:
        st.caption(
            "Every LLM-extracted relationship stores the sentence it was based "
            "on. This is the difference between a knowledge graph and a rumour "
            "graph: an edge you cannot trace back to a sentence is an edge you "
            "cannot defend."
        )
        evidence = detail.get("evidence", [])
        if not evidence:
            st.info("No relationships to show evidence for.")
        for row in evidence:
            provenance = row.get("provenance", "erp")
            st.markdown(
                f"**{row['type']}** → {row['neighbour']} "
                + pill("model-extracted" if provenance == "llm" else "system of record",
                       "llm" if provenance == "llm" else "erp")
                + f'<span class="muted">confidence {row.get("confidence", 1.0):.2f}</span>',
                unsafe_allow_html=True,
            )
            if row.get("evidence"):
                st.markdown(
                    f'<div class="evidence">"{row["evidence"]}"<br>'
                    f'<span class="muted">- {row.get("source_doc", "")}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="evidence muted">From the structured export, '
                    'where the record itself is the evidence.</div>',
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        docs = detail.get("documents", [])
        if docs:
            st.dataframe(pd.DataFrame(docs), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No documents mention this entity.")


# ===========================================================================
# Ask
# ===========================================================================
def render_answer(payload: dict, show_trace: bool = True) -> None:
    validation = payload.get("validation", {})
    if validation and not validation.get("ok", True):
        st.warning("Output guardrails flagged this answer")
        for warning in validation.get("warnings", []):
            st.markdown(
                pill(warning["severity"], "block" if warning["severity"] == "error"
                     else "warn") + f" **{warning['kind']}** - {warning['detail']}",
                unsafe_allow_html=True,
            )

    st.markdown(payload["answer"])

    metrics = payload.get("metrics", {})
    cols = st.columns(6)
    cols[0].metric("Chunks", metrics.get("text_chunks", 0))
    cols[1].metric("Graph facts", metrics.get("graph_facts", 0))
    cols[2].metric("Entities", metrics.get("entities", 0))
    cols[3].metric("Max hops", metrics.get("max_hops", 0))
    cols[4].metric("Latency", f"{metrics.get('total_ms', 0):.0f} ms")
    cols[5].metric("Cost", f"${payload.get('usage', {}).get('estimated_usd', 0):.5f}")

    retrieval = payload.get("retrieval", {})

    if show_trace and retrieval.get("trace"):
        with st.expander("How the answer was retrieved, step by step", expanded=True):
            for i, line in enumerate(retrieval["trace"], 1):
                st.markdown(f"**{i}.** {line}")

    graph_facts = [e for e in retrieval.get("evidence", [])
                   if e["kind"] == "graph_fact"]
    if graph_facts:
        with st.expander(
            f"{len(graph_facts)} derived graph fact(s) - computed by traversal, "
            "quoted from no document", expanded=True,
        ):
            for item in graph_facts:
                st.code(item["text"], language="text")

    if retrieval.get("entities"):
        with st.expander(f"{len(retrieval['entities'])} entities reached by traversal"):
            st.dataframe(
                pd.DataFrame([
                    {"entity": e["name"], "type": e["type"], "hops": e["hops"],
                     "path": " → ".join(e.get("path_names") or [])}
                    for e in retrieval["entities"]
                ]),
                use_container_width=True, hide_index=True,
            )

    text_evidence = [e for e in retrieval.get("evidence", []) if e["kind"] == "text"]
    if text_evidence:
        with st.expander(f"{len(text_evidence)} text chunks retrieved"):
            for item in text_evidence:
                st.markdown(
                    f"**`{item['doc_id']}`** · {item['title']} · found by "
                    f"*{item['retrieved_by']}* · score {item['score']:.4f}"
                )
                st.markdown(
                    f'<div class="evidence muted">{item["text"][:600]}'
                    f'{"..." if len(item["text"]) > 600 else ""}</div>',
                    unsafe_allow_html=True,
                )

    if retrieval.get("cypher_run"):
        with st.expander("The Cypher that actually ran"):
            st.caption(
                "Read from `src/graph/queries.py` at run time, so this is the "
                "query itself and not a prettified copy."
            )
            for cypher in retrieval["cypher_run"]:
                st.code(cypher.strip(), language="cypher")

    if payload.get("guardrails", {}).get("checks"):
        with st.expander("Guardrail checks on this request"):
            st.dataframe(pd.DataFrame(payload["guardrails"]["checks"]),
                         use_container_width=True, hide_index=True)


def tab_ask(client: ApiClient) -> None:
    st.header("Ask a question")

    if not client.census().get("populated"):
        st.warning("The graph is empty. Build it on the **Build** tab first.")
        return

    strategies = client.strategies()
    ids = [s["id"] for s in strategies]

    preset = st.selectbox("Try one of these", ["(write my own)"] + SAMPLE_QUESTIONS)
    default = "" if preset == "(write my own)" else preset
    question = st.text_area("Question", value=default, height=90,
                            key=f"ask_{hash(default)}")

    strategy = st.radio(
        "Retrieval strategy", ids, index=ids.index("hybrid"), horizontal=True,
        format_func=lambda s: next(x["label"] for x in strategies if x["id"] == s),
    )
    st.caption(next(x["description"] for x in strategies if x["id"] == strategy))

    if st.button("Answer", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving and generating"):
            try:
                payload = client.ask(question, strategy)
            except ApiError as exc:
                show_api_error(exc)
                return
        render_answer(payload)


# ===========================================================================
# Compare
# ===========================================================================
def tab_compare(client: ApiClient) -> None:
    st.header("Compare retrieval strategies")
    st.markdown(
        "The most important tab. GraphRAG is **not** universally better than "
        "RAG, and a system that claims otherwise is selling something. Run the "
        "same question through every strategy and read the differences."
    )

    if not client.census().get("populated"):
        st.warning("The graph is empty. Build it on the **Build** tab first.")
        return

    strategies = client.strategies()
    ids = [s["id"] for s in strategies]

    with st.expander("What each strategy does"):
        st.dataframe(pd.DataFrame(strategies), use_container_width=True,
                     hide_index=True)

    preset = st.selectbox("Question", SAMPLE_QUESTIONS, key="cmp_preset")
    question = st.text_area("Question", value=preset, height=90, key="cmp_q")
    chosen = st.multiselect("Strategies", ids, default=ids,
                            format_func=lambda s: next(
                                x["label"] for x in strategies if x["id"] == s))

    if st.button("Run comparison", type="primary",
                 disabled=not question.strip() or not chosen):
        with st.spinner(f"Running {len(chosen)} strategies"):
            try:
                payload = client.compare(question, chosen)
            except ApiError as exc:
                show_api_error(exc)
                return

        st.subheader("At a glance")
        st.dataframe(pd.DataFrame(payload["comparison"]),
                     use_container_width=True, hide_index=True)
        st.caption(
            "Read the cost columns as carefully as the quality ones. A strategy "
            "that uses twice the context and twice the latency to reach the "
            "same answer has lost."
        )

        if payload.get("document_matrix"):
            st.subheader("Which documents each strategy found")
            st.caption(
                "Usually the clearest single view of *why* one strategy wins - "
                "it shows the actual evidence gap rather than an aggregate score."
            )
            st.dataframe(pd.DataFrame(payload["document_matrix"]),
                         use_container_width=True, hide_index=True)

        st.subheader("Answers")
        for result in payload["results"]:
            label = result["retrieval"]["label"]
            with st.expander(label, expanded=result["strategy"] in ("hybrid", "vector")):
                render_answer(result)


# ===========================================================================
# Cypher
# ===========================================================================
def tab_cypher(client: ApiClient) -> None:
    st.header("Schema and Cypher")
    st.markdown(
        "In a graph project the data model *is* the architecture, so it is "
        "worth looking at directly rather than through the application."
    )

    census = client.census()
    if not census.get("populated"):
        st.warning("The graph is empty. Build it on the **Build** tab first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Node labels")
        st.dataframe(
            pd.DataFrame([{"label": k, "count": v}
                          for k, v in census["nodes"].items()]),
            use_container_width=True, hide_index=True,
        )
    with col2:
        st.subheader("Relationship types")
        st.dataframe(
            pd.DataFrame([{"type": k, "count": v}
                          for k, v in census["relationships"].items()]),
            use_container_width=True, hide_index=True,
        )

    schema = client.schema()
    st.subheader("Indexes and constraints, as the database reports them")
    st.caption(
        "Not as the code claims them. The difference matters: a vector index "
        "that failed to come online returns no rows and no error. A uniqueness "
        "constraint also creates a backing index, so these are the lookup path "
        "every MERGE in ingestion seeks on - without them ingestion goes "
        "quadratic."
    )
    st.dataframe(pd.DataFrame(schema["indexes"]), use_container_width=True,
                 hide_index=True)

    st.divider()
    st.subheader("Cookbook")
    cookbook = client.cookbook()
    names = [entry["name"] for entry in cookbook]
    choice = st.selectbox("Query", names)
    entry = next(e for e in cookbook if e["name"] == choice)

    st.info(entry["explanation"])
    st.code(entry["cypher"], language="cypher")
    if entry["parameters"]:
        st.caption(f"Parameters: `{entry['parameters']}`")
    if st.button("Run it", type="primary"):
        try:
            result = client.run_cypher(entry["cypher"], entry["parameters"])
        except ApiError as exc:
            show_api_error(exc)
        else:
            if result["rows"]:
                st.dataframe(pd.DataFrame(result["rows"]),
                             use_container_width=True, hide_index=True)
                st.caption(f"{result['row_count']} rows in "
                           f"{result['elapsed_ms']:.1f} ms")
            else:
                st.warning("No rows returned.")

    with st.expander("Write your own (read-only)"):
        st.caption(
            "Write operations are rejected by the API. That is not a security "
            "boundary - anyone who can reach the API can reach the database - "
            "it guards against destroying your own graph with a stray DETACH "
            "DELETE mid-lesson. The real boundary is Neo4j RBAC with a "
            "read-only role, described in docs/security.md."
        )
        custom = st.text_area(
            "Cypher", height=130,
            value="MATCH (s:Supplier)-[r:DEPENDS_ON]->(u:Supplier)\n"
                  "RETURN s.name AS supplier, u.name AS upstream,\n"
                  "       r.confidence AS confidence, r.source_doc AS source",
        )
        if st.button("Run query"):
            try:
                result = client.run_cypher(custom)
            except ApiError as exc:
                show_api_error(exc)
            else:
                st.dataframe(
                    pd.DataFrame(result["rows"]) if result["rows"] else pd.DataFrame(),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"{result['row_count']} rows in "
                           f"{result['elapsed_ms']:.1f} ms")


# ===========================================================================
# Guardrails
# ===========================================================================
def tab_guardrails(client: ApiClient) -> None:
    st.header("Guardrails and security")
    st.markdown(
        "GraphRAG has an injection problem that ordinary RAG does not, and it "
        "is worse in a specific, structural way."
    )

    st.info(
        "**In ordinary RAG, a poisoned document corrupts one answer.**\n\n"
        "In GraphRAG the same document is processed by an extractor whose "
        "output is **written to shared, persistent state**. A sentence crafted "
        "to read as a supply relationship becomes an *edge* that persists "
        "indefinitely, is reached by traversals from unrelated questions, "
        "affects every user, and arrives in later answers presented as a "
        "*derived graph fact* - carrying a real citation, because the sentence "
        "genuinely does appear in a genuine document.\n\n"
        "Every downstream groundedness check passes. The claim is grounded - in "
        "a lie someone planted."
    )

    try:
        config = client.guardrail_config()
    except ApiError as exc:
        show_api_error(exc)
        return

    st.subheader("Enforcement points")
    st.dataframe(pd.DataFrame(config["enforcement_points"]),
                 use_container_width=True, hide_index=True)

    with st.expander("Active policy"):
        st.json({k: v for k, v in config.items() if k != "enforcement_points"})

    st.divider()
    st.subheader("Try it")
    st.caption(
        "Scan any text. `as document` includes the graph-poisoning patterns, "
        "which apply only to text destined for the extractor - a user may "
        "legitimately ask \"which suppliers should we add a second source "
        "for\", while the same words in an ingested document are an instruction "
        "aimed at the thing that writes to the database. Context decides "
        "severity."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load the adversarial sample"):
            try:
                st.session_state["scan_text"] = client.adversarial_sample()["text"]
            except ApiError as exc:
                show_api_error(exc)
    with col2:
        as_document = st.toggle("Scan as an ingested document", value=True)

    text = st.text_area("Text to scan", height=220,
                        value=st.session_state.get("scan_text", ""),
                        key="scan_box")

    if st.button("Scan", type="primary", disabled=not text.strip()):
        try:
            result = client.scan(text, as_document=as_document)
        except ApiError as exc:
            show_api_error(exc)
        else:
            if result["blocked"]:
                st.error(f"BLOCKED - {result['summary']}")
            elif result["needs_review"]:
                st.warning(f"Flagged for review - {result['summary']}")
            else:
                st.success("Clean - no patterns matched")

            if result["detections"]:
                st.dataframe(pd.DataFrame(result["detections"]),
                             use_container_width=True, hide_index=True)

            pii = result["pii"]
            st.markdown(f"**PII / secrets:** {pii['summary']}")
            if pii["findings"]:
                st.dataframe(pd.DataFrame(pii["findings"]),
                             use_container_width=True, hide_index=True)
                st.caption(
                    "Excerpts are truncated on purpose. Logging a credential to "
                    "prove you found a credential is a bug that has caused real "
                    "incidents."
                )

    st.divider()
    st.subheader("Audit log")
    st.caption(
        "Append-only, JSON lines. Not compliance theatre: it is what lets you "
        "reconstruct why the system said something three weeks later, find "
        "which ingestion introduced a bad edge, and notice a guardrail that "
        "quietly stopped working after a refactor."
    )
    try:
        audit = client.audit(limit=200)
    except ApiError as exc:
        show_api_error(exc)
        return

    stats = audit["stats"]
    cols = st.columns(3)
    cols[0].metric("Events", stats["events_read"])
    cols[1].metric("Blocked", stats["blocked"])
    cols[2].metric("With warnings", stats["with_warnings"])
    if audit["events"]:
        st.dataframe(pd.DataFrame(audit["events"]), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No events yet. Ask a question and come back.")

    st.caption(
        "Detection is best-effort and always will be - a determined attacker "
        "writes around any pattern list. **Traceability is not.** Every "
        "extracted edge stores its provenance, its confidence and the verbatim "
        "sentence it came from, so a bad edge can be found and proven rather "
        "than argued about. Run `python run.py security` for the full "
        "demonstration."
    )


# ===========================================================================
# Evaluate
# ===========================================================================
def tab_evaluate(client: ApiClient) -> None:
    st.header("Evaluation")
    st.markdown(
        "Everything this project claims is measured against a fixed set of "
        "golden questions. The set deliberately includes questions GraphRAG "
        "should **lose** - a benchmark containing only questions your system "
        "wins is marketing, not evaluation."
    )

    try:
        questions = client.golden_questions()
    except ApiError as exc:
        show_api_error(exc)
        return

    st.subheader(f"The question set ({len(questions)} questions)")
    st.dataframe(
        pd.DataFrame([
            {"id": q["id"], "category": q["category"],
             "expected winner": q["expected_advantage"],
             "hops": q["hops_required"],
             "needs graph fact": q.get("needs_graph_fact", False),
             "question": q["question"]}
            for q in questions
        ]),
        use_container_width=True, hide_index=True,
    )

    with st.expander("What each metric means, and what a good result looks like"):
        st.markdown(
            "- **evidence recall** - of the documents a correct answer needs, "
            "how many reached the context. Isolates retrieval from generation: "
            "if this is 0, no prompt work will save the answer.\n"
            "- **term coverage** - of the terms a correct answer must contain, "
            "how many appear. Deterministic string matching, no judge, "
            "identical on every run. A proxy, not a grade.\n"
            "- **graph fact rate** - how often a derived traversal fact reached "
            "the context. The direct test of the core claim.\n"
            "- **faithfulness** - judged 0–1. The one metric here needing a "
            "model, and the one to read most sceptically: the judge shares a "
            "model family with the system it grades.\n"
            "- **refusal** - on the two genuinely unanswerable questions, did it "
            "decline instead of inventing. Higher is better.\n"
            "- **unwarranted refusal** - on *answerable* questions, how often it "
            "said it could not tell. That is a retrieval failure wearing a "
            "polite hat.\n\n"
            "**A good result** has hybrid at or near 1.0 recall on multi-hop, "
            "vector *matching* hybrid on single-document and definitional "
            "questions, and graph-only clearly losing somewhere. If the graph "
            "strategies win every category, the baseline is broken and every "
            "other number is worthless."
        )

    st.divider()
    status = client.eval_status()
    if status.get("running"):
        st.info("Evaluation is running.")
        bar = st.progress(status.get("progress", 0.0),
                          text=status.get("message", "working"))
        while status.get("running"):
            time.sleep(2)
            status = client.eval_status()
            bar.progress(min(status.get("progress", 0.0), 1.0),
                         text=status.get("message", "working"))
        st.rerun()

    strategies = [s["id"] for s in client.strategies()]
    subset = st.multiselect("Questions (empty = all)", [q["id"] for q in questions],
                            default=[q["id"] for q in questions[:3]])
    chosen = st.multiselect("Strategies", strategies, default=strategies)
    judge = st.checkbox("Use the LLM judge (doubles the calls)", value=False)
    st.caption(
        "The full run is 12 questions × 5 strategies, about seven minutes with "
        "the judge on. Prefer `python run.py eval` in a terminal for the "
        "complete benchmark; this is for spot checks."
    )

    if st.button("Run evaluation", type="primary", disabled=not chosen):
        try:
            client.start_eval(strategies=chosen, question_ids=subset or None,
                              judge=judge)
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)

    if status.get("error"):
        st.error(status["error"])

    report = status.get("report")
    if report:
        st.subheader("By strategy")
        st.dataframe(
            pd.DataFrame(report["by_strategy"]).T.reset_index(names="strategy"),
            use_container_width=True, hide_index=True,
        )
        st.subheader("By category - the table that matters")
        st.dataframe(
            pd.DataFrame(report["by_category"]).T.reset_index(
                names="category / strategy"),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"{report['questions']} questions, {report['seconds']}s, "
                   f"${report['usage']['estimated_usd']:.4f}")


# ===========================================================================
def main() -> None:
    client = get_client()

    ok, error = client.reachable()
    if not ok:
        st.title("GraphRAG Supply Chain Intelligence")
        st.error("The backend is not reachable")
        st.code(error or "unknown error", language="text")
        st.markdown(
            "This UI is a **thin client**: all retrieval, Cypher, prompting, "
            "guardrail and scoring logic runs in the FastAPI backend, so it "
            "cannot answer anything on its own. That is deliberate - a "
            "guardrail enforced in a frontend is a guardrail anyone can skip "
            "with curl.\n\n"
            "Start the backend in another terminal:\n"
            "```bash\npython run.py api\n```"
        )
        st.stop()

    sidebar(client)

    tabs = st.tabs(["Build", "Explore", "Ask", "Compare", "Cypher",
                    "Guardrails", "Evaluate"])
    renderers = [tab_build, tab_explore, tab_ask, tab_compare, tab_cypher,
                 tab_guardrails, tab_evaluate]
    for tab, renderer in zip(tabs, renderers):
        with tab:
            try:
                renderer(client)
            except ApiError as exc:
                show_api_error(exc)


main()
