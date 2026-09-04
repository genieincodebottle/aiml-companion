"""Grounded answer generation.

The retrieval layer decides *what the model sees*.  This layer decides *what
the model is allowed to do with it*, and the second matters as much as the
first.  A perfect retrieval feeding a careless prompt still produces confident
fabrication.

Three properties are enforced here:

  GROUNDING       The model may use only the supplied context.  Its own
                  knowledge of supply chains, however correct in general, is
                  not evidence about Northwind.
  ATTRIBUTION     Every factual sentence carries a citation, and derived graph
                  facts are cited as derived.  A reader must be able to tell
                  "the audit report says X" from "the graph computed X".
  REFUSAL         When the context does not contain the answer, saying so is
                  the correct output.  This is the single most important line
                  in the prompt, and the reason the evaluation measures
                  refusal behaviour explicitly rather than treating every
                  non-answer as a failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .guardrails import GuardrailEngine
from .llm import LLMClient
from .retrieval.base import RetrievalResult

SYSTEM_PROMPT = """\
You are a supply chain risk analyst at Northwind Instruments. You answer \
questions using ONLY the evidence provided to you.

Rules:

1. Use only the supplied context. If the context does not support an answer, \
say exactly what is missing. Never fill a gap with general knowledge about \
supply chains, manufacturing or the companies named.
2. Cite every factual claim. Cite a document as [DOC-ID]. Cite a derived fact \
as [graph: <id>], using the id shown on the block.
3. Blocks marked DERIVED FROM THE KNOWLEDGE GRAPH were computed by traversing \
relationships, not quoted from a document. Treat them as reliable structural \
facts, and when you use one, say briefly which chain of relationships it \
rests on so the reader can check it.
4. Distinguish what the evidence states from what it implies. If you draw a \
conclusion the evidence supports but does not state, mark it as an inference.
5. Be concise and specific. Prefer naming the actual suppliers, components, \
products and locations over describing categories of risk.
6. If the evidence is partial, answer the part you can and state plainly what \
you could not determine and what would be needed to determine it.
"""

USER_TEMPLATE = """\
Question: {question}

Evidence:

{context}

Answer the question using only the evidence above."""


@dataclass
class Answer:
    question: str
    strategy: str
    text: str
    retrieval: RetrievalResult
    context_chars: int = 0
    generation_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    # What the output guardrails found. Returned alongside the answer rather
    # than hidden, because a guardrail the user cannot see is one they cannot
    # trust - and because a flagged answer is still the best evidence of how
    # the system misbehaved.
    validation: dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.retrieval.latency_ms + self.generation_ms

    @property
    def cited_documents(self) -> list[str]:
        """Which document ids the answer actually cited.

        Compared against what was retrieved, this is a cheap and surprisingly
        informative signal: a large gap between retrieved and cited means the
        retriever is padding the context with material the model found useless,
        which is latency and money spent for nothing.
        """
        import re
        found = set(re.findall(r"\[([A-Z0-9\-]{3,})\]", self.text))
        available = {e.doc_id for e in self.retrieval.text_evidence if e.doc_id}
        return sorted(found & available)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "strategy": self.strategy,
            "answer": self.text,
            "cited_documents": self.cited_documents,
            "context_chars": self.context_chars,
            "validation_ok": self.validation.get("ok", True),
            "validation_warnings": len(self.validation.get("warnings", [])),
            "retrieval_ms": round(self.retrieval.latency_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            **self.retrieval.summary(),
        }


class AnswerEngine:
    def __init__(self, llm: LLMClient, config: Config,
                 guard: GuardrailEngine | None = None) -> None:
        self.llm = llm
        self.config = config
        self.guard = guard or GuardrailEngine(config)

    def answer(self, question: str, retrieval: RetrievalResult,
               graph_entity_names: list[str] | None = None) -> Answer:
        max_chars = self.config.retrieval["max_context_chars"]
        context = retrieval.context(max_chars)

        if not context.strip():
            # Do not spend a model call to be told there is nothing to say.
            # This also keeps the refusal wording deterministic, which the
            # evaluation depends on.
            return Answer(
                question=question, strategy=retrieval.strategy,
                text=("Nothing was retrieved for this question, so there is no "
                      "evidence to answer from. This is a retrieval failure, "
                      "not an absence of information in the corpus."),
                retrieval=retrieval, context_chars=0,
            )

        prompt = USER_TEMPLATE.format(question=question, context=context)
        before = self.llm.usage.as_dict()
        started = time.perf_counter()
        text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        elapsed = (time.perf_counter() - started) * 1000
        after = self.llm.usage.as_dict()

        _decision, validation = self.guard.check_answer(
            text, context=context,
            available_documents={e.doc_id for e in retrieval.text_evidence if e.doc_id},
            graph_entity_names=graph_entity_names,
            graph_fact_ids=[e.source_id for e in retrieval.graph_evidence],
        )

        return Answer(
            question=question,
            strategy=retrieval.strategy,
            text=text,
            retrieval=retrieval,
            context_chars=len(context),
            generation_ms=elapsed,
            validation=validation.as_dict(),
            usage={
                "input_tokens": after["input_tokens"] - before["input_tokens"],
                "output_tokens": after["output_tokens"] - before["output_tokens"],
                "estimated_usd": round(after["estimated_usd"] - before["estimated_usd"], 6),
            },
        )
