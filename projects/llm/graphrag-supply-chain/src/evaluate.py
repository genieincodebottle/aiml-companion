"""Evaluation: measuring whether the claims in the README are true.

WHAT WE MEASURE, AND WHY EACH ONE

  evidence_recall     Of the documents a correct answer needs, how many did
                      retrieval actually put in the context?
                      This isolates retrieval from generation. If recall is 0,
                      no prompt engineering will save the answer, and time
                      spent on the prompt is time wasted.

  graph_fact_present  Did the context include a derived (traversal) fact?
                      Only meaningful for questions marked needs_graph_fact.
                      This is the direct test of the project's core claim: that
                      some answers exist only as a join.

  term_coverage       Of the terms a correct answer must contain, how many
                      appear? Deterministic string matching, no model, no
                      judge, identical on every run.
                      It is a proxy, not a grade: an answer could contain every
                      term and still be wrong. It is used because it is
                      reproducible, and it is reported alongside a judged
                      score rather than instead of one.

  forbidden_hits      Terms whose presence marks a specific known error, such
                      as listing a product that has no battery under a battery
                      regulation. A precision check to sit against recall.

  refused_correctly   For the two unanswerable questions, did the system
                      decline instead of inventing? Measured by a judge, since
                      refusal has no fixed wording.

  faithfulness        Judged 0-1: is every claim in the answer supported by
                      the supplied context? This is the hallucination measure
                      and it is the one metric here that needs a model.

WHAT A GOOD RESULT LOOKS LIKE

  evidence_recall 1.0 on multi_hop questions for graph and hybrid, and clearly
  below that for vector - if vector also scores 1.0, the questions are not
  actually multi-hop and the benchmark is broken.

  vector at least matching graph on single_document and definitional. If the
  graph strategies win those too, something is wrong with the baseline, and a
  rigged baseline invalidates every other number on the page.

ON THE JUDGE

  faithfulness and refusal are scored by the same model family that wrote the
  answers. That is a real limitation: a model is a lenient judge of its own
  output, and these numbers should be read as directional. The deterministic
  metrics above are not subject to that, which is why both are reported.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answer import AnswerEngine
from .config import Config, get_config
from .graph.client import GraphClient
from .llm import LLMClient
from .retrieval.strategies import STRATEGIES, Retriever

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithfulness": {
            "type": "number",
            "description": "0 to 1. Fraction of factual claims in the answer that are supported by the context.",
        },
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "refused": {
            "type": "boolean",
            "description": "True if the answer declines to answer or states the information is unavailable.",
        },
    },
    "required": ["faithfulness", "unsupported_claims", "refused"],
}

JUDGE_SYSTEM = """\
You grade an answer against the evidence it was given. You are strict.

A claim is supported only if the context states it or it follows directly from \
the context by arithmetic or by reading a listed relationship. General \
knowledge is never support. Plausibility is never support.

Set `refused` true only if the answer declines to state a fact it does not \
have, or explicitly says the information is not in the evidence. An answer \
that hedges but still asserts the fact has not refused.
"""

JUDGE_TEMPLATE = """\
Question: {question}

Context given to the answering system:
{context}

Answer produced:
{answer}

Grade it."""


@dataclass
class QuestionResult:
    question_id: str
    strategy: str
    category: str
    evidence_recall: float
    term_coverage: float
    forbidden_hits: list[str]
    graph_fact_present: bool
    faithfulness: float | None
    refused: bool | None
    must_refuse: bool
    unsupported_claims: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    total_ms: float = 0.0
    context_chars: int = 0
    answer: str = ""
    retrieved_documents: list[str] = field(default_factory=list)

    @property
    def refusal_correct(self) -> bool | None:
        if self.refused is None:
            return None
        return self.refused == self.must_refuse

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "strategy": self.strategy,
            "category": self.category,
            "evidence_recall": round(self.evidence_recall, 3),
            "term_coverage": round(self.term_coverage, 3),
            "forbidden_hits": self.forbidden_hits,
            "graph_fact_present": self.graph_fact_present,
            "faithfulness": None if self.faithfulness is None else round(self.faithfulness, 3),
            "refused": self.refused,
            "must_refuse": self.must_refuse,
            "refusal_correct": self.refusal_correct,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "context_chars": self.context_chars,
            "retrieved_documents": self.retrieved_documents,
        }


def load_questions(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["questions"]


def _coverage(text: str, terms: list[str]) -> tuple[float, list[str]]:
    if not terms:
        return 1.0, []
    lowered = text.lower()
    hits = [t for t in terms if t.lower() in lowered]
    return len(hits) / len(terms), [t for t in terms if t.lower() not in lowered]


def evaluate(*, strategies: list[str] | None = None, judge: bool = True,
             question_ids: list[str] | None = None,
             config: Config | None = None,
             progress: Any = None) -> dict[str, Any]:
    config = config or get_config()
    strategies = strategies or list(STRATEGIES)
    questions = load_questions(config.golden_questions)
    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]

    llm = LLMClient(config)
    results: list[QuestionResult] = []
    started = time.time()

    with GraphClient(config) as client:
        client.verify()
        retriever = Retriever(client, llm, config)
        engine = AnswerEngine(llm, config)

        # Warm the embedding cache for every question BEFORE timing anything.
        #
        # This is a correction to a real flaw in the first version of this
        # harness. Strategies run in a fixed order, and `vector` runs first, so
        # it paid the ~500 ms embedding API round trip while `classic` and
        # `hybrid` - which embed the identical question - hit the disk cache and
        # measured ~30 ms. The published table therefore showed dense retrieval
        # as 16x slower than a strategy that does strictly more work, including
        # the same embedding call.
        #
        # That is not a small distortion, it is a reversal, and it would have
        # been invisible in the output. Pre-warming makes every strategy measure
        # the same thing: graph and index work, with the embedding round trip
        # excluded from all of them equally. What the latency column now
        # compares is the cost of the retrieval logic itself.
        for question in questions:
            llm.embed_query(question["question"])

        total = len(questions) * len(strategies)
        done = 0
        for question in questions:
            for strategy in strategies:
                retrieval = retriever.retrieve(question["question"], strategy)
                answer = engine.answer(question["question"], retrieval)

                retrieved_docs = sorted(
                    {e.doc_id for e in retrieval.text_evidence if e.doc_id}
                )
                required = question.get("required_documents") or []
                recall = (
                    len([d for d in required if d in retrieved_docs]) / len(required)
                    if required else 1.0
                )
                coverage, _missing = _coverage(answer.text, question.get("required_terms") or [])
                _, _ = _coverage(answer.text, [])
                forbidden = [
                    t for t in (question.get("forbidden_terms") or [])
                    if t.lower() in answer.text.lower()
                ]

                faithfulness: float | None = None
                refused: bool | None = None
                unsupported: list[str] = []
                if judge:
                    verdict = _judge(llm, question["question"],
                                     retrieval.context(config.retrieval["max_context_chars"]),
                                     answer.text)
                    if verdict:
                        faithfulness = float(verdict.get("faithfulness") or 0.0)
                        refused = bool(verdict.get("refused"))
                        unsupported = list(verdict.get("unsupported_claims") or [])

                results.append(
                    QuestionResult(
                        question_id=question["id"],
                        strategy=strategy,
                        category=question["category"],
                        evidence_recall=recall,
                        term_coverage=coverage,
                        forbidden_hits=forbidden,
                        graph_fact_present=bool(retrieval.graph_evidence),
                        faithfulness=faithfulness,
                        refused=refused,
                        must_refuse=bool(question.get("must_refuse")),
                        unsupported_claims=unsupported,
                        retrieval_ms=retrieval.latency_ms,
                        total_ms=answer.total_ms,
                        context_chars=answer.context_chars,
                        answer=answer.text,
                        retrieved_documents=retrieved_docs,
                    )
                )
                done += 1
                if progress:
                    progress(f"{question['id']} / {strategy}", done / total)

    return {
        "results": [r.as_dict() for r in results],
        "by_strategy": _aggregate(results, key=lambda r: r.strategy),
        "by_category": _aggregate(
            results, key=lambda r: f"{r.category}::{r.strategy}"
        ),
        "usage": llm.usage.as_dict(),
        "seconds": round(time.time() - started, 1),
        "questions": len(questions),
        "strategies": strategies,
        "judged": judge,
        "answers": {f"{r.question_id}::{r.strategy}": r.answer for r in results},
    }


def _aggregate(results: list[QuestionResult], key: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[QuestionResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)

    out: dict[str, dict[str, Any]] = {}
    for name, items in sorted(groups.items()):
        judged = [i for i in items if i.faithfulness is not None]
        refusals = [i for i in items if i.refusal_correct is not None]
        out[name] = {
            "n": len(items),
            "evidence_recall": round(statistics.fmean(i.evidence_recall for i in items), 3),
            "term_coverage": round(statistics.fmean(i.term_coverage for i in items), 3),
            "graph_fact_rate": round(
                statistics.fmean(1.0 if i.graph_fact_present else 0.0 for i in items), 3
            ),
            "forbidden_hits": sum(len(i.forbidden_hits) for i in items),
            "faithfulness": (
                round(statistics.fmean(i.faithfulness or 0.0 for i in judged), 3)
                if judged else None
            ),
            # Two separate numbers, because one number here hides the more
            # interesting half. `refusal_accuracy` asks: on the questions the
            # corpus genuinely cannot answer, did we decline? `unwarranted_
            # refusal` asks the opposite: on answerable questions, how often did
            # we say we could not tell? The second is not a safety failure, it
            # is a RETRIEVAL failure wearing a polite hat - the model behaved
            # correctly given evidence that never arrived.
            "refusal_accuracy": (
                round(statistics.fmean(
                    1.0 if i.refused else 0.0
                    for i in refusals if i.must_refuse), 3)
                if any(i.must_refuse for i in refusals) else None
            ),
            "unwarranted_refusal": (
                round(statistics.fmean(
                    1.0 if i.refused else 0.0
                    for i in refusals if not i.must_refuse), 3)
                if any(not i.must_refuse for i in refusals) else None
            ),
            "median_retrieval_ms": round(statistics.median(i.retrieval_ms for i in items), 1),
            "median_total_ms": round(statistics.median(i.total_ms for i in items), 1),
            "median_context_chars": int(statistics.median(i.context_chars for i in items)),
        }
    return out


def _judge(llm: LLMClient, question: str, context: str, answer: str) -> dict[str, Any] | None:
    if not answer.strip():
        return {"faithfulness": 0.0, "unsupported_claims": [], "refused": True}
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context=context[:20000] or "(no context was retrieved)",
        answer=answer,
    )
    return llm.extract_json(prompt, JUDGE_SCHEMA, system=JUDGE_SYSTEM)


def format_report(report: dict[str, Any]) -> str:
    """A terminal-readable summary.  The JSON is the record; this is the part a
    human reads without opening a file."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"GraphRAG evaluation: {report['questions']} questions x "
                 f"{len(report['strategies'])} strategies "
                 f"({report['seconds']}s, judged={report['judged']})")
    lines.append("=" * 78)
    lines.append("")

    header = (f"{'strategy':<10} {'recall':>7} {'terms':>7} {'graph':>7} "
              f"{'faith':>7} {'refuse':>7} {'unwarr':>7} {'bad':>5} "
              f"{'ret ms':>8} {'ctx':>7}")
    lines.append("OVERALL, ALL QUESTIONS")
    lines.append(header)
    lines.append("-" * len(header))
    for name, row in report["by_strategy"].items():
        lines.append(
            f"{name:<10} {row['evidence_recall']:>7.3f} {row['term_coverage']:>7.3f} "
            f"{row['graph_fact_rate']:>7.3f} "
            f"{_fmt(row['faithfulness']):>7} {_fmt(row['refusal_accuracy']):>7} "
            f"{_fmt(row['unwarranted_refusal']):>7} "
            f"{row['forbidden_hits']:>5} {row['median_retrieval_ms']:>8.1f} "
            f"{row['median_context_chars']:>7}"
        )

    lines.append("")
    lines.append("BY QUESTION CATEGORY  (this is the table that matters)")
    lines.append(f"{'category / strategy':<38} {'recall':>7} {'terms':>7} {'graph':>7} {'ms':>8}")
    lines.append("-" * 70)
    for name, row in report["by_category"].items():
        lines.append(
            f"{name:<38} {row['evidence_recall']:>7.3f} {row['term_coverage']:>7.3f} "
            f"{row['graph_fact_rate']:>7.3f} {row['median_retrieval_ms']:>8.1f}"
        )
    lines.append("")
    lines.append("recall = required documents retrieved | terms = required answer terms present")
    lines.append("graph  = share of questions where a derived traversal fact reached the context")
    lines.append("refuse = declined on the 2 genuinely unanswerable questions (higher is better)")
    lines.append("unwarr = declined on an ANSWERABLE question, i.e. retrieval failed (lower is better)")
    lines.append("ret ms = retrieval only, embedding cache pre-warmed for every strategy equally")
    lines.append("")
    lines.append(f"Estimated cost of this run: ${report['usage']['estimated_usd']:.4f} "
                 f"({report['usage']['llm_calls']} LLM calls)")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> int:
    report = evaluate()
    print(format_report(report))
    out = get_config().root / "artifacts" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nFull results (including every answer) written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
