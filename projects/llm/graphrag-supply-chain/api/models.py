"""Request and response schemas.

These are the contract between the backend and any client, and they are the
reason the Streamlit app can contain no business logic: everything it needs
arrives already shaped, already validated, already guarded.

Pydantic is doing real work here beyond documentation. `Field` constraints
reject an oversized question or an out-of-range hop count at the edge of the
system, before a request reaches the retriever - which means the guardrail for
that class of input is enforced by the framework and cannot be forgotten by a
route handler.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Strategy = Literal["vector", "keyword", "classic", "graph", "hybrid"]


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000,
        description="Natural-language question about the supply chain.",
    )
    strategy: Strategy = Field(
        "hybrid", description="Retrieval strategy to use.",
    )
    include_trace: bool = Field(
        True, description="Include the step-by-step retrieval narration.",
    )


class EvidenceOut(BaseModel):
    kind: str
    text: str
    source_id: str
    doc_id: str = ""
    title: str = ""
    doc_type: str = ""
    score: float = 0.0
    retrieved_by: str = ""


class GraphEntityOut(BaseModel):
    key: str
    name: str
    type: str
    hops: int = 0
    path_names: list[str] = Field(default_factory=list)
    path_rels: list[str] = Field(default_factory=list)


class RetrievalOut(BaseModel):
    strategy: str
    label: str
    trace: list[str] = Field(default_factory=list)
    cypher_run: list[str] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    entities: list[GraphEntityOut] = Field(default_factory=list)
    latency_ms: float = 0.0
    stats: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    strategy: str
    answer: str
    retrieval: RetrievalOut
    metrics: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    # What the guardrails decided, always returned. A guardrail the client
    # cannot see is a guardrail the user cannot trust.
    guardrails: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    strategies: list[Strategy] = Field(
        default_factory=lambda: ["vector", "keyword", "classic", "graph", "hybrid"],
        min_length=1, max_length=5,
    )


class CompareResponse(BaseModel):
    question: str
    results: list[AskResponse]
    comparison: list[dict[str, Any]]
    document_matrix: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class CensusResponse(BaseModel):
    nodes: dict[str, int]
    relationships: dict[str, int]
    total_nodes: int
    total_relationships: int
    populated: bool


class EntitySummary(BaseModel):
    key: str
    name: str
    type: str


class EntityDetail(BaseModel):
    key: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    summary: str | None = None
    status: str | None = None
    neighbours: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class SubgraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    seeds: list[str]


class CypherRequest(BaseModel):
    """Ad-hoc read-only query.

    The write-blocking below is not a security boundary - anyone who can reach
    this API can reach the database - it is a guard against a learner
    destroying their own graph with a stray DETACH DELETE mid-lesson. The real
    boundary is Neo4j's own role-based access control, which a production
    deployment configures with a read-only user for this endpoint.
    """

    cypher: str = Field(..., min_length=6, max_length=4000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class CypherResponse(BaseModel):
    rows: list[dict[str, Any]]
    row_count: int
    elapsed_ms: float


class CookbookEntry(BaseModel):
    name: str
    cypher: str
    parameters: dict[str, Any]
    explanation: str


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    reset: bool = Field(True, description="Wipe the graph before rebuilding.")


class IngestStatus(BaseModel):
    running: bool
    message: str = ""
    progress: float = 0.0
    report: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class EvalRequest(BaseModel):
    strategies: list[Strategy] | None = None
    question_ids: list[str] | None = None
    judge: bool = False


class EvalStatus(BaseModel):
    running: bool
    message: str = ""
    progress: float = 0.0
    report: dict[str, Any] | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class ScanRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    as_document: bool = Field(
        True,
        description=("Scan as an ingested document (includes the graph-poisoning "
                     "patterns) rather than as a user question."),
    )


class ScanResponse(BaseModel):
    blocked: bool
    needs_review: bool
    summary: str
    detections: list[dict[str, Any]]
    pii: dict[str, Any]
