"""Question-answering service: the orchestration layer for ask and compare.

LAYERING
========
    app/            UI. Renders. Decides nothing.
    api/routes_*    Routing. Validates the request shape, calls one service
                    method, maps the result to a response model. No domain
                    rules, no orchestration, no aggregation.
    src/services/   THIS LAYER. Orchestration and policy: which guardrail runs
                    when, what a comparison means, what gets audited.
    src/            Capabilities. Retrieval, graph, LLM, guardrails - each does
                    one thing and knows nothing about HTTP.

Why the service layer exists at all, when the routes could just call the
retriever: because "answer a question" is not one call. It is a guardrail
check, then retrieval, then generation, then output validation, then a budget
check, then an audit write - in that order, with specific failure handling at
each step. That sequence IS the business logic. Putting it in a route handler
means it exists only for callers who arrive over HTTP, so the CLI, the
notebook and the tests each grow their own slightly different copy, and the
copies drift.

Services raise domain exceptions (`GuardrailViolation`, `BudgetExceeded`). They
never raise `HTTPException` and never import from `fastapi`, which is what
keeps them callable from a notebook and testable without a web server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..answer import Answer, AnswerEngine
from ..config import Config
from ..guardrails import GuardrailDecision, GuardrailEngine
from ..guardrails.limits import BudgetExceeded
from ..llm import LLMClient
from ..retrieval.base import RetrievalResult
from ..retrieval.strategies import STRATEGY_LABELS, Retriever


@dataclass
class AnswerBundle:
    """One answered question, with everything the caller needs to render it."""

    answer: Answer
    retrieval: RetrievalResult
    usage: dict[str, Any]
    guardrails: GuardrailDecision | None = None

    @property
    def label(self) -> str:
        return STRATEGY_LABELS.get(self.retrieval.strategy, self.retrieval.strategy)


@dataclass
class ComparisonBundle:
    question: str
    bundles: list[AnswerBundle] = field(default_factory=list)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    document_matrix: list[dict[str, Any]] = field(default_factory=list)


class QAService:
    def __init__(self, *, config: Config, retriever: Retriever,
                 answers: AnswerEngine, guard: GuardrailEngine,
                 llm: LLMClient, entity_names: Any) -> None:
        self.config = config
        self.retriever = retriever
        self.answers = answers
        self.guard = guard
        self.llm = llm
        # A callable rather than a list: the entity set changes on ingestion,
        # and a snapshot captured at construction would go stale and start
        # flagging newly-added, entirely real suppliers as hallucinations.
        self._entity_names = entity_names

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, strategy: str, *, caller: str = "local",
            include_trace: bool = True) -> AnswerBundle:
        """The full pipeline for one question.

        Raises GuardrailViolation or BudgetExceeded. Both are domain errors the
        caller is expected to translate for its own transport.
        """
        started = time.perf_counter()

        # 1. Query-time guardrails: rate limit, length, injection, secrets.
        decision = self.guard.check_question(question, caller)

        # 2-4. Retrieve, generate, validate. Note the GUARDED text is used, not
        # the original - scanning one string and then using another is a
        # guardrail that reports success and protects nothing.
        bundle = self._answer(decision.text, strategy, caller)
        bundle.guardrails = decision
        if not include_trace:
            bundle.retrieval.trace = []

        # 5. Audit. Written for every request, not only failures: a guardrail
        # nobody can see firing is one that silently stops working.
        self.guard.audit.write(
            "ask", caller=caller, strategy=strategy,
            question_chars=len(question),
            documents=sorted({e.doc_id for e in bundle.retrieval.text_evidence
                              if e.doc_id}),
            graph_facts=len(bundle.retrieval.graph_evidence),
            validation_ok=bundle.answer.validation.get("ok", True),
            warnings=[w["kind"] for w in bundle.answer.validation.get("warnings", [])],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return bundle

    def _answer(self, question: str, strategy: str, caller: str) -> AnswerBundle:
        before = self.llm.usage.as_dict()

        retrieval = self.retriever.retrieve(question, strategy)
        answer = self.answers.answer(
            question, retrieval, graph_entity_names=self._entity_names()
        )

        after = self.llm.usage.as_dict()
        usage = {
            "llm_calls": after["llm_calls"] - before["llm_calls"],
            "input_tokens": after["input_tokens"] - before["input_tokens"],
            "output_tokens": after["output_tokens"] - before["output_tokens"],
            "estimated_usd": round(after["estimated_usd"] - before["estimated_usd"], 6),
        }

        # Budget checked after the fact. Checking before would require
        # predicting token counts, which cannot be done accurately; checking
        # after still catches a runaway before the next request compounds it,
        # and records the breach either way.
        try:
            self.guard.budget.check(usage, "request")
        except BudgetExceeded:
            self.guard.audit.write("budget_exceeded", caller=caller,
                                   strategy=strategy, usage=usage, blocked=True)
            raise

        return AnswerBundle(answer=answer, retrieval=retrieval, usage=usage)

    # -------------------------------------------------------------- compare
    def compare(self, question: str, strategies: list[str], *,
                caller: str = "local") -> ComparisonBundle:
        decision = self.guard.check_question(question, caller)

        # Warm the embedding cache before timing anything.
        #
        # Without this the comparison is dishonest in a way that is invisible in
        # the output: strategies run in order, the first to embed pays the API
        # round trip, and every later strategy hits the disk cache. That made
        # dense retrieval look 16x slower than a strategy doing strictly more
        # work, including the same embedding call. Warming first means every
        # strategy is measured on its retrieval logic, which is the thing being
        # compared.
        self.llm.embed_query(decision.text)

        bundles = [self._answer(decision.text, strategy, caller)
                   for strategy in strategies]

        self.guard.audit.write("compare", caller=caller, strategies=strategies,
                               question_chars=len(question))

        return ComparisonBundle(
            question=question,
            bundles=bundles,
            comparison=[self._summarise(b) for b in bundles],
            document_matrix=self._document_matrix(bundles),
        )

    @staticmethod
    def _summarise(bundle: AnswerBundle) -> dict[str, Any]:
        metrics = bundle.answer.as_dict()
        return {
            "strategy": bundle.retrieval.strategy,
            "label": bundle.label,
            "chunks": metrics.get("text_chunks", 0),
            "graph_facts": metrics.get("graph_facts", 0),
            "entities": metrics.get("entities", 0),
            "max_hops": metrics.get("max_hops", 0),
            "documents": metrics.get("documents", 0),
            "context_chars": metrics.get("context_chars", 0),
            "retrieval_ms": metrics.get("retrieval_ms", 0),
            "total_ms": metrics.get("total_ms", 0),
            "estimated_usd": bundle.usage.get("estimated_usd", 0),
            "validation_ok": bundle.answer.validation.get("ok", True),
        }

    @staticmethod
    def _document_matrix(bundles: list[AnswerBundle]) -> list[dict[str, Any]]:
        """Which documents each strategy found that the others did not.

        Usually the single clearest view of *why* one strategy wins - more so
        than any aggregate score, because it shows the actual evidence gap
        rather than a number derived from it.
        """
        doc_sets = {
            b.retrieval.strategy: {e.doc_id for e in b.retrieval.text_evidence
                                   if e.doc_id}
            for b in bundles
        }
        if not doc_sets:
            return []
        every = sorted(set().union(*doc_sets.values()))
        return [
            {"document": doc, **{s: (doc in docs) for s, docs in doc_sets.items()}}
            for doc in every
        ]

    # -------------------------------------------------------------- catalogue
    @staticmethod
    def strategy_catalogue() -> list[dict[str, str]]:
        descriptions = {
            "vector": "Embed the question, take the k nearest chunks. The "
                      "textbook RAG baseline.",
            "keyword": "BM25 over chunk text. Strong on part numbers and codes, "
                       "which embeddings are bad at.",
            "classic": "Vector + BM25 fused with Reciprocal Rank Fusion. The "
                       "honest baseline - comparing GraphRAG against dense-only "
                       "would rig the experiment.",
            "graph": "Entity linking, traversal, derived facts, then supporting "
                     "text. No vector search at all, so it fails when the "
                     "question names nothing in the graph.",
            "hybrid": "Vector anchors, walk MENTIONS into the graph, traverse, "
                      "then fuse everything. The full architecture.",
        }
        return [
            {"id": key, "label": label, "description": descriptions[key]}
            for key, label in STRATEGY_LABELS.items()
        ]
