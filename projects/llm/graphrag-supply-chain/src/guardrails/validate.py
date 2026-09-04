"""Output guardrails: checking the answer before the user sees it.

Retrieval guardrails decide what the model may read. These decide what it may
say. They run after generation and before the response is returned.

Three checks, all deterministic and none requiring a model:

  CITATIONS   Every [DOC-ID] the answer cites must be a document that was
              actually in the context. A citation to a document the model never
              saw is a fabricated citation, and it is the most damaging failure
              mode in this whole system - it produces an answer that *looks*
              auditable and is not.

  ENTITIES    Every supplier, component or product named in the answer should
              exist in the graph. A confident sentence about "Pacific Rim
              Components" when no such supplier exists is a hallucination that
              a groundedness judge often misses, because the sentence around it
              is perfectly well-formed.

  NUMBERS     Figures in the answer should be traceable to the context. Models
              are notably willing to adjust a number slightly while copying it,
              and "14 weeks of magnet inventory" becoming "40 weeks" is the kind
              of error that survives review because nobody re-reads the source.

WHY DETERMINISTIC CHECKS AND NOT JUST AN LLM JUDGE
==================================================
Because the judge shares a failure mode with the thing it is judging. A model
that finds "Pacific Rim Components" plausible when writing the answer finds it
equally plausible when grading it. String matching against the actual retrieved
context and the actual graph has no such correlation - it is checking a fact
about the world, not asking for a second opinion.

These checks WARN rather than block by default. Blocking on a numeric mismatch
would suppress correct answers that performed arithmetic the check cannot
follow ("three of the four products"). The right default is to surface the
warning next to the answer so a human can look, and to let the caller decide
whether a given deployment should be stricter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_CITATION = re.compile(r"\[([A-Z][A-Z0-9\-]{2,})\]")
_GRAPH_CITATION = re.compile(r"\[graph:\s*([^\]]+)\]")

# Candidate entity mentions: two or more consecutive capitalised words. Crude,
# and tuned to over-collect: the check that follows filters against the graph,
# so a false candidate costs nothing while a missed one costs a hallucination.
# Hyphens are inside the token class on purpose. An earlier version used
# [A-Z][a-z]{2,}, which could not see "Pan-Asia Laminate Group" as one name -
# the hyphen broke the \s+ between words, so a fabricated supplier slipped
# through the check written to catch exactly that. Found by `run.py security`.
_CANDIDATE_ENTITY = re.compile(r"\b([A-Z][\w-]{2,}(?:\s+[A-Z][\w-]{2,}){1,3})\b")

# Words that end an organisation name. Requiring one keeps the false-positive
# rate usable: without it, ordinary capitalised prose ("Supply Risk Bulletin")
# is flagged constantly, the warning gets ignored, and an ignored warning is
# worse than no warning at all.
#
# The cost is a known false-negative class: a fabricated supplier carrying no
# corporate suffix ("Northgate Laminate") is not flagged. That is a deliberate
# precision/recall trade, and the honest description of this control is "it
# catches the common shape of an invented company name" - not "it detects
# hallucinated entities".
_ORG_SUFFIX = re.compile(
    r"\b(Ltd|Limited|Inc|Incorporated|Corp|Corporation|Company|GmbH|AG|BV|NV|"
    r"AB|SA|SRL|Pte|Pty|Sdn|Bhd|Pvt|Group|Holdings|Partners|Associates|"
    r"Industries|Technologies|Solutions|Systems|Materials|Components|"
    r"Electronics|Manufacturing|Works|Refining|Precision|Chemicals|Polymers|"
    r"Plastics|Metals|Circuits|Optics|Sensing|Fluidics|Foil|Glass|Salts|"
    r"Laminate|Substrate|Semiconductor|Energi|Energy|Cell|Cells)\b"
)

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent)?")

# Words that begin a capitalised phrase without naming an entity.
_ENTITY_STOPWORDS = {
    "The", "This", "That", "These", "Those", "It", "They", "We", "Our", "Note",
    "However", "Because", "Since", "Although", "While", "Based", "According",
    "Evidence", "Answer", "Question", "Derived", "Graph", "Knowledge", "Supply",
    "Chain", "Risk", "Both", "Each", "Every", "There", "What", "Which",
}

# Numbers that are almost never a claim about the world.
_TRIVIAL_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                    "100", "2024", "2025", "2026"}


@dataclass
class Warning_:
    kind: str
    detail: str
    severity: str = "warn"        # 'warn' | 'error'


@dataclass
class ValidationResult:
    warnings: list[Warning_] = field(default_factory=list)
    cited_documents: list[str] = field(default_factory=list)
    fabricated_citations: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(w.severity == "error" for w in self.warnings)

    @property
    def clean(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        if not self.warnings:
            return "passed all output checks"
        return "; ".join(f"{w.kind}: {w.detail}" for w in self.warnings)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "clean": self.clean,
            "cited_documents": self.cited_documents,
            "fabricated_citations": self.fabricated_citations,
            "unknown_entities": self.unknown_entities,
            "ungrounded_numbers": self.ungrounded_numbers,
            "warnings": [{"kind": w.kind, "detail": w.detail, "severity": w.severity}
                         for w in self.warnings],
        }


def _normalise_number(raw: str) -> str:
    return raw.replace(",", "").rstrip("0").rstrip(".") if "." in raw else raw.replace(",", "")


def validate_answer(answer: str, *, context: str,
                    available_documents: Iterable[str],
                    graph_entity_names: Iterable[str] | None = None,
                    known_graph_fact_ids: Iterable[str] | None = None,
                    ) -> ValidationResult:
    result = ValidationResult()
    available = {d.upper() for d in available_documents}
    context_lower = context.lower()

    # -- 1. citations ------------------------------------------------------
    cited = {m.group(1).upper() for m in _CITATION.finditer(answer)}
    # A citation-shaped token that is not a document id is usually the model
    # citing an entity or a part number in brackets. Only treat it as a
    # fabricated citation if it looks like a document id, meaning it contains a
    # hyphen and is not a known part number in the context.
    fabricated = sorted(
        token for token in cited - available
        if "-" in token and token.lower() not in context_lower
    )
    result.cited_documents = sorted(cited & available)
    result.fabricated_citations = fabricated
    if fabricated:
        result.warnings.append(
            Warning_(
                kind="fabricated_citation",
                detail=(f"cites {', '.join(fabricated)}, which was not in the "
                        "retrieved context"),
                # An error, not a warning. An answer that cites a document it
                # never saw is worse than an uncited one, because it presents as
                # verified.
                severity="error",
            )
        )

    known_facts = {f.lower() for f in (known_graph_fact_ids or [])}
    if known_facts:
        for match in _GRAPH_CITATION.finditer(answer):
            ref = match.group(1).strip().lower()
            if ref not in known_facts:
                result.warnings.append(
                    Warning_(
                        kind="fabricated_graph_citation",
                        detail=f"cites graph fact '{ref}' that was not derived",
                        severity="error",
                    )
                )

    # -- 2. entities -------------------------------------------------------
    if graph_entity_names is not None:
        known = {name.lower() for name in graph_entity_names}
        # Also allow anything that literally appears in the context: the answer
        # may correctly name something the corpus mentions but the graph never
        # made a node for.
        candidates = {
            m.group(1) for m in _CANDIDATE_ENTITY.finditer(answer)
            if m.group(1).split()[0] not in _ENTITY_STOPWORDS
        }
        unknown = sorted(
            c for c in candidates
            if c.lower() not in known and c.lower() not in context_lower
            # Require a corporate-looking suffix; see _ORG_SUFFIX above for the
            # precision/recall trade this makes and what it therefore misses.
            and _ORG_SUFFIX.search(c)
        )
        result.unknown_entities = unknown
        if unknown:
            result.warnings.append(
                Warning_(
                    kind="unknown_entity",
                    detail=(f"names {', '.join(unknown)}, which exists in "
                            "neither the graph nor the retrieved text"),
                    severity="error",
                )
            )

    # -- 3. numbers --------------------------------------------------------
    context_numbers = {
        _normalise_number(m.group(1)) for m in _NUMBER.finditer(context)
    }
    ungrounded: list[str] = []
    for match in _NUMBER.finditer(answer):
        raw = match.group(1)
        value = _normalise_number(raw)
        if value in _TRIVIAL_NUMBERS or value in context_numbers:
            continue
        ungrounded.append(raw + (match.group(2) or ""))
    result.ungrounded_numbers = sorted(set(ungrounded))
    if result.ungrounded_numbers:
        result.warnings.append(
            Warning_(
                kind="ungrounded_number",
                detail=(f"states {', '.join(result.ungrounded_numbers)}, which "
                        "does not appear in the retrieved context (may be "
                        "arithmetic the check cannot follow)"),
                severity="warn",
            )
        )

    return result
