"""Prompt-injection defence, at both ends of the pipeline.

READ THIS SECTION EVEN IF YOU SKIP THE CODE
===========================================

GraphRAG has an injection problem that ordinary RAG does not, and it is worse
in a specific, structural way.

In a normal RAG system, a poisoned document corrupts **one answer**. The
attacker's text lands in a context window, the model does something it should
not, the user gets a bad response, and the next question starts clean.

In GraphRAG, the same poisoned document is processed by an **extractor whose
output is written to shared, persistent state**. A sentence crafted to read as
a supply relationship becomes an *edge*. That edge:

  - persists after the attack, in the database, indefinitely;
  - is reached by traversals from questions that have nothing to do with the
    poisoned document;
  - affects **every user**, not just the one who asked;
  - arrives in future answers laundered as a "derived graph fact", which this
    system's own prompt instructs the model to treat as a reliable structural
    fact;
  - and carries a real citation, because the evidence sentence genuinely does
    appear in a genuine document.

That last property is what makes it dangerous. Every downstream check for
groundedness passes. The claim is grounded - in a lie someone planted.

The threat is not hypothetical in this domain. Supplier questionnaires, audit
responses and certificates are documents that **outside parties author and send
you**, which is the textbook precondition for indirect prompt injection. A
supplier who wants to look less concentrated than they are has a motive to
write "Meridian Circuits also sources laminate from three qualified
alternatives" into a questionnaire response.

So there are two defences here, and they defend different things:

  scan_document()  runs at INGESTION, before the extractor. It protects the
                   INTEGRITY OF THE GRAPH. This is the important one.
  scan_question()  runs at QUERY TIME, before retrieval. It protects one
                   answer and the system prompt.

And one that matters more than either: `src/ingest/extract.py` requires a
verbatim evidence quote for every relationship, and `provenance` on every edge
records that it came from a model. Detection is best-effort and always will be;
**traceability is not**. The realistic security posture for LLM extraction is
not "we caught every injection" but "every claim can be traced to a sentence in
a named document, so when something is wrong we can find it and prove it".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Patterns.
#
# These are heuristics, and they are documented as heuristics. A determined
# attacker writes around any keyword list, and anyone who tells you their regex
# stops prompt injection is selling something. What a list like this genuinely
# buys you is coverage of opportunistic and copy-pasted attacks, which is most
# of what actually arrives, plus a signal for human review.
#
# The patterns are grouped by what they indicate, because the response differs:
# an instruction override in a supplier PDF is close to conclusive evidence of
# tampering, while an unusual amount of imperative mood is merely worth a look.
# ---------------------------------------------------------------------------

INSTRUCTION_OVERRIDE = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|preceding|earlier)\s+"
    r"(instruction|prompt|rule|direction|context)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"forget\s+(everything|all|what)\s+(you|above|before)",
    r"new\s+(instruction|system\s+prompt|rule)s?\s*[:\-]",
    r"override\s+(your|the|all)\s+(instruction|rule|setting|guardrail)",
    r"you\s+are\s+now\s+(a|an|the)\b",
    r"from\s+now\s+on[,\s]+(you|always|never)\b",
]

ROLE_HIJACK = [
    r"^\s*(system|assistant|developer)\s*[:\-]",
    r"<\|?\s*(system|im_start|im_end|endoftext)\s*\|?>",
    r"\[\s*(system|inst|/inst)\s*\]",
    r"###\s*(system|instruction)s?\b",
    r"act\s+as\s+(a|an|the)\s+\w+\s+(and|then)\b",
]

EXFILTRATION = [
    r"(reveal|print|show|repeat|output|disclose)\s+(your|the)\s+"
    r"(system\s+)?(prompt|instruction|rule|configuration|api\s+key|secret)",
    r"what\s+(were|are)\s+your\s+(original\s+)?instructions",
    r"repeat\s+(everything|the\s+text)\s+above",
]

# GraphRAG-specific. These target the EXTRACTOR rather than the chat model, and
# they are the ones a generic prompt-injection filter will not be looking for.
GRAPH_POISONING = [
    r"(add|create|insert|record|register)\s+(a\s+|the\s+|an\s+)?"
    r"(new\s+)?(relationship|edge|node|entity|dependency|supplier)\b",
    r"(extract|emit|output|return)\s+(the\s+)?following\s+"
    r"(relationship|entity|triple|edge|json)",
    r"when\s+(asked|queried|answering)\s+about\b.{0,60}\b(say|state|respond|answer)\b",
    r"do\s+not\s+(extract|record|report|include|mention)\b",
    r"mark\s+(this|the)\s+\w+\s+as\s+(low\s+risk|safe|compliant|approved)",
    r"confidence\s*[:=]\s*1\.0",          # trying to hand-set an edge weight
]

_GROUPS: dict[str, list[str]] = {
    "instruction_override": INSTRUCTION_OVERRIDE,
    "role_hijack": ROLE_HIJACK,
    "exfiltration": EXFILTRATION,
    "graph_poisoning": GRAPH_POISONING,
}

_COMPILED = {
    name: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    for name, patterns in _GROUPS.items()
}

# Severity per group. `block` stops the document or question outright; `review`
# lets it through but records it and surfaces it to a human.
_SEVERITY = {
    "instruction_override": "block",
    "role_hijack": "block",
    "exfiltration": "block",
    "graph_poisoning": "review",
}

# Zero-width and bidirectional control characters. These are invisible when a
# human reviews the document and fully visible to the tokeniser, which is
# exactly the asymmetry an attacker wants. There is no legitimate reason for
# U+202E (right-to-left override) to appear in a supplier audit report.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")


@dataclass
class Detection:
    group: str
    severity: str
    pattern: str
    excerpt: str


@dataclass
class ScanResult:
    ok: bool
    detections: list[Detection] = field(default_factory=list)
    cleaned: str = ""

    @property
    def blocked(self) -> bool:
        return any(d.severity == "block" for d in self.detections)

    @property
    def needs_review(self) -> bool:
        return any(d.severity == "review" for d in self.detections)

    def summary(self) -> str:
        if not self.detections:
            return "clean"
        parts = sorted({f"{d.group}({d.severity})" for d in self.detections})
        return ", ".join(parts)


def _scan(text: str, groups: Iterable[str]) -> list[Detection]:
    found: list[Detection] = []
    for group in groups:
        for pattern in _COMPILED[group]:
            match = pattern.search(text)
            if match:
                start = max(0, match.start() - 60)
                end = min(len(text), match.end() + 60)
                found.append(
                    Detection(
                        group=group,
                        severity=_SEVERITY[group],
                        pattern=pattern.pattern[:60],
                        excerpt=text[start:end].replace("\n", " ").strip(),
                    )
                )
    return found


def strip_invisible(text: str) -> tuple[str, int]:
    """Replace zero-width and bidi control characters with a SPACE.

    Always applied, never merely flagged. Unlike the keyword heuristics this
    one has no false-positive cost: legitimate business documents do not
    contain right-to-left overrides.

    SUBSTITUTE, DO NOT DELETE - and this is not a style preference, it is the
    whole point of the function.

    An earlier version deleted them, which handed the attacker exactly what
    they wanted. The payload

        Ignore<ZWSP>previous<ZWSP>instructions

    reads to a human as one nonsense word and to the tokeniser as the
    instruction. Deleting the zero-width characters produced
    "Ignorepreviousinstructions", which matches no pattern containing a whitespace class,
    so the sanitiser DEFEATED the detector: the text got cleaner and the attack
    got through. Substituting a space restores the word boundaries the attacker
    was hiding, and the pattern matches.

    This was caught by `python run.py security`, which is the argument for
    demonstrating controls against real payloads instead of asserting them.
    """
    cleaned, count = _INVISIBLE.subn(" ", text)
    return cleaned, count


def scan_document(text: str, doc_id: str = "") -> ScanResult:
    """Ingestion-time scan. Protects the integrity of the graph.

    Runs every pattern group, including the graph-poisoning set, because this
    text is about to be fed to an extractor whose output becomes persistent
    shared state.
    """
    cleaned, invisible = strip_invisible(text)
    detections = _scan(cleaned, _GROUPS)
    if invisible:
        detections.append(
            Detection(
                group="invisible_characters", severity="review",
                pattern="zero-width/bidi control characters",
                excerpt=f"{invisible} invisible characters removed from {doc_id or 'document'}",
            )
        )
    blocked = any(d.severity == "block" for d in detections)
    return ScanResult(ok=not blocked, detections=detections, cleaned=cleaned)


def scan_question(text: str) -> ScanResult:
    """Query-time scan. Protects one answer and the system prompt.

    The graph-poisoning group is *not* applied here, and that is deliberate. A
    user legitimately asks "which suppliers should we add a second source for",
    and a pattern tuned to catch "add a new supplier relationship" in a document
    would reject it. The same string means different things depending on whether
    it arrives in a question or in a document destined for the extractor -
    context decides severity, which is why there are two functions and not one.
    """
    cleaned, _ = strip_invisible(text)
    detections = _scan(
        cleaned, ["instruction_override", "role_hijack", "exfiltration"]
    )
    blocked = any(d.severity == "block" for d in detections)
    return ScanResult(ok=not blocked, detections=detections, cleaned=cleaned)


def wrap_untrusted(text: str, source: str) -> str:
    """Delimit retrieved content so the model can tell data from instruction.

    This is a mitigation, not a fix, and it is worth being precise about why.
    Delimiters help because they give the model a clear frame; they do not
    *guarantee* anything, because the model has no enforced separation between
    the two channels. Anyone claiming otherwise has not read the literature.

    It is still worth doing: combined with an explicit instruction in the system
    prompt that content inside the markers is data, it measurably reduces
    success rates for opportunistic attacks, at a cost of a few tokens.
    """
    return (
        f"<untrusted_document source=\"{source}\">\n"
        f"{text}\n"
        f"</untrusted_document>"
    )
