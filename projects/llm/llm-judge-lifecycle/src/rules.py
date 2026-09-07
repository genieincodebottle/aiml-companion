"""A deterministic rule engine that reads the same rubric an LLM judge reads.

Why this exists
---------------
A rubric in this project is markdown prose written for a model to read. Some
bullets additionally carry an inline tag:

    - Reject filler that would fit any item in the catalogue.
      [banned: "you'll love it", "a must-watch", "highly rated"]
    - Every factual claim must be traceable to the record. [grounded]
    - Name the item. [require: subject.title]

An LLM judge ignores the brackets and reads the sentence. This engine ignores
the sentence and reads the brackets. **Same rubric, two readers.**

That is not a trick to make an offline demo work. It buys three things:

1. **A free baseline arm.** Before you can claim an LLM judge is worth its
   latency and its bill, you have to beat the rules you could have written
   instead. Most published judge results skip this comparison. Here it is one
   config line: set ``judge.provider: stub``. On the reference domain the rule
   engine is genuinely competitive on two of the three criteria, and that
   result is the most useful thing a beginner can learn from this repo.

2. **Phase II works with no API key.** RART's optimiser edits rubric text. If
   the evaluator reads that text, the loop is real: adding a banned phrase
   measurably changes alignment against the held-out human labels. Nothing is
   scripted, nothing is faked, and a learner can watch the objective move
   before spending anything. The algorithm is byte-for-byte the one that runs
   against Gemini; only the reader of the rubric differs.

3. **Hermetic tests.** Determinism means the test suite asserts on numbers.

The honest limits, which the README repeats
-------------------------------------------
Substring rules cannot read. They miss paraphrase ("a film you are certain to
adore" sails past a ban on "you'll love it"), they cannot tell a supported
inference from an invented one, and they have no idea what the item is about.
That ceiling is the argument FOR an LLM judge, and this project is built so you
can measure the gap yourself rather than take anyone's word for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Tag vocabulary. Small on purpose: every kind here has to be something a human
# would also write as a plain sentence, or the two readers stop agreeing about
# what the rubric says.
#
# `spoiler` and `sensitive` are deliberately separate kinds rather than one
# `unsafe` bucket. They map to different failure modes, and that distinction is
# not cosmetic: reasoning agreement is scored by comparing the mode the judge
# names against the mode the human named. Collapse the two and every safety
# catch scores as a reason mismatch - the judge reached the right verdict and
# reported a reason too coarse to be checked. `unsafe` is kept as the generic
# fallback for domains that genuinely have only one kind of unsafe.
KINDS = (
    "banned",
    "require",
    "grounded",
    "max_words",
    "min_words",
    "unsafe",
    "spoiler",
    "sensitive",
)

# Kinds whose arguments are substrings matched against the artefact.
_PHRASE_KINDS = frozenset({"banned", "unsafe", "spoiler", "sensitive"})

# kind -> (failure_mode, reason template)
_SAFETY_KINDS = {
    "spoiler": (
        "spoiler",
        "Gives away how the story resolves ({hit!r}). A recommendation that "
        "spoils the thing it is recommending has destroyed the product it is "
        "selling, and for that viewer it cannot be undone.",
    ),
    "sensitive": (
        "sensitive_framing",
        "Uses difficult subject matter as an enticement ({hit!r}) rather than "
        "as a description.",
    ),
    "unsafe": (
        "unsafe_content",
        "Contains wording flagged as unsafe to show ({hit!r}).",
    ),
}

_TAG_RE = re.compile(r"\[(\w+)(?::\s*(.*?))?\]", re.DOTALL)

# DOUBLE quotes only, and this is not a stylistic choice.
#
# Accepting the single quote as a delimiter too looks more permissive and is
# actively broken, because the phrases most worth banning contain apostrophes.
# `[banned: "you'll love it", "a must-watch"]` parsed under a permissive regex
# yields the fragments `you`, `,` and `,` - and a one-character banned phrase
# matches the comma in almost every well-formed sentence in the corpus.
#
# The symptom is what made this worth a comment: nothing raised, no tag was
# reported as malformed, and the rule engine simply rejected every artefact
# containing a comma. Recall collapsed to 0.29 while specificity stayed
# plausible, so it read as a judge that was merely too strict rather than as a
# parser bug. Silent, plausible, and wrong in the direction that looks like a
# tuning problem.
_QUOTED_RE = re.compile(r'"([^"]+)"')

# Words that start a sentence or are simply common enough that their
# capitalisation says nothing about whether they name a thing.
_CAP_STOPWORDS = frozenset(
    """
    a an the and or but if then this that these those it its it's they them their
    you your we our i he she his her him hers as at by for from in into of on to
    with without about over under after before during while because so than
    there here when where what which who whom whose how why all any both each
    few more most other some such no nor not only own same too very can will
    just should now also like unlike think thought perfect ideal great good best
    if though although unless until since despite yet still even
    """.split()
)


@dataclass(frozen=True)
class Rule:
    kind: str
    args: tuple[Any, ...] = ()
    source: str = ""

    def describe(self) -> str:
        if self.args:
            return f"{self.kind}({', '.join(str(a) for a in self.args)})"
        return self.kind


@dataclass
class RuleVerdict:
    label: str  # "PASS" | "FAIL"
    reason: str
    failure_mode: str | None = None
    triggered: list[str] = field(default_factory=list)


def parse_rubric(rubric: str) -> list[Rule]:
    """Pull the machine-readable tags out of a rubric.

    Unknown tags are skipped rather than raising. The reflector writes rubrics,
    and a reflector that invents ``[vibes: good]`` should degrade this engine to
    "one fewer rule", not crash a tuning run at iteration 4 and lose every
    completed evaluation in it.
    """
    rules: list[Rule] = []
    for match in _TAG_RE.finditer(rubric or ""):
        kind = match.group(1).strip().lower()
        if kind not in KINDS:
            continue
        raw = (match.group(2) or "").strip()
        rules.append(Rule(kind=kind, args=_parse_args(kind, raw), source=match.group(0)))
    return rules


def _parse_args(kind: str, raw: str) -> tuple[Any, ...]:
    if not raw:
        return ()
    if kind in ("max_words", "min_words"):
        try:
            return (int(raw),)
        except ValueError:
            return ()
    quoted = _QUOTED_RE.findall(raw)
    args = (
        [q.strip().lower() for q in quoted if q.strip()]
        if quoted
        else [part.strip().lower() for part in raw.split(",") if part.strip()]
    )

    if kind in _PHRASE_KINDS:
        # A phrase rule is a substring match, so a one- or two-character
        # argument matches nearly every sentence in the corpus. That is never
        # what anyone meant, and the damage is silent - it rejects everything
        # while the metrics merely look like an over-strict judge. Drop it and
        # say so, because a rubric with one dropped clause is recoverable and a
        # rubric that rejects the whole corpus is a day of debugging.
        kept = [a for a in args if len(a) >= 3]
        if len(kept) != len(args):
            import logging

            logging.getLogger(__name__).warning(
                "rubric tag [%s: ...] contained fragment(s) %s shorter than 3 "
                "characters; dropping them. Phrases must be wrapped in DOUBLE "
                "quotes - a single quote is not a delimiter here, because the "
                "phrases worth banning contain apostrophes.",
                kind,
                [a for a in args if len(a) < 3],
            )
        args = kept

    return tuple(args)


class RuleEngine:
    """Applies parsed rules to one artefact against one record.

    Ordering matters and is deliberate: safety first, then groundedness, then
    specificity, then length. When several rules fire, the reported failure mode
    is the most serious one - which is what a human rater does too, and what
    makes the reasoning-agreement metric meaningful rather than a coin flip
    between two true statements.
    """

    def __init__(self, rubric: str) -> None:
        self.rubric = rubric
        self.rules = parse_rubric(rubric)

    def evaluate(self, artefact: str, record: dict[str, Any]) -> RuleVerdict:
        text = (artefact or "").strip()
        low = text.lower()
        words = text.split()
        support = _support_corpus(record)
        triggered: list[str] = []

        for kind, (mode, template) in _SAFETY_KINDS.items():
            for rule in self._of(kind):
                hits = [t for t in rule.args if t in low]
                if hits:
                    return RuleVerdict(
                        "FAIL", template.format(hit=hits[0]), mode, [rule.describe()]
                    )

        if self._of("grounded"):
            unsupported = _unsupported_claims(text, support)
            if unsupported:
                return RuleVerdict(
                    "FAIL",
                    "Makes a claim the record does not support: "
                    f"{unsupported[0]!r}. Every factual detail has to be "
                    "traceable to the record.",
                    "unsupported_claim",
                    ["grounded"],
                )
            triggered.append("grounded")

        for rule in self._of("require"):
            for path in rule.args:
                value = _dig(record, str(path))
                if value and str(value).lower() not in low:
                    return RuleVerdict(
                        "FAIL",
                        f"Never names {value!r}, so the reader cannot tell which "
                        "item this is about.",
                        "missing_subject",
                        [rule.describe()],
                    )

        for rule in self._of("banned"):
            hits = [p for p in rule.args if p in low]
            if hits:
                return RuleVerdict(
                    "FAIL",
                    f"Generic filler: {hits[0]!r} would be equally true of any "
                    "item in the catalogue, so it carries no information.",
                    "generic_filler",
                    [rule.describe()],
                )

        for rule in self._of("max_words"):
            if rule.args and len(words) > rule.args[0]:
                return RuleVerdict(
                    "FAIL",
                    f"Runs to {len(words)} words against a limit of "
                    f"{rule.args[0]}.",
                    "too_long",
                    [rule.describe()],
                )

        for rule in self._of("min_words"):
            if rule.args and len(words) < rule.args[0]:
                return RuleVerdict(
                    "FAIL",
                    f"Only {len(words)} words, below the floor of {rule.args[0]}; "
                    "too thin to justify anything.",
                    "too_short",
                    [rule.describe()],
                )

        return RuleVerdict("PASS", "Satisfies every clause of the rubric.", None, triggered)

    def _of(self, kind: str) -> list[Rule]:
        return [r for r in self.rules if r.kind == kind]

    def banned_phrases(self) -> set[str]:
        return {str(a) for r in self._of("banned") for a in r.args}


# ---------------------------------------------------------------------------
# Support corpus and claim checking
# ---------------------------------------------------------------------------


def _support_corpus(record: dict[str, Any]) -> str:
    """Everything the artefact is allowed to assert, flattened to one string.

    A claim is "supported" if it appears here. Crude - an artefact could recite
    the record without understanding it - but it catches the failure that
    matters most in practice: a fluent, confident detail that is simply not in
    the source.
    """
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif node is not None and not isinstance(node, bool):
            parts.append(str(node))

    for key in ("subject", "references", "facts", "context"):
        if key in record:
            walk(record[key])
    return " ".join(parts).lower()


def _unsupported_claims(text: str, support: str) -> list[str]:
    """Numbers and named entities in the artefact that the record never mentions.

    Numbers are checked because they are the highest-confidence hallucination
    signal in short copy - a model that writes "rated 4.8 by 12,000 people" when
    the record holds neither figure has invented both, and a reader has no way
    to tell.

    Named entities are approximated by capitalised token runs that are not
    sentence-initial. That approximation is wrong sometimes in both directions,
    which is exactly why this engine is a baseline and not the destination.
    """
    out: list[str] = []

    for number in re.findall(r"\b\d[\d,.]*%?\b", text):
        if number.strip(".,").lower() not in support:
            out.append(number)

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        tokens = sentence.split()
        for index, token in enumerate(tokens):
            if index == 0:
                continue  # sentence-initial capitalisation means nothing
            bare = token.strip(".,;:!?\"'()")
            if not bare or not bare[0].isupper():
                continue
            if bare.lower() in _CAP_STOPWORDS:
                continue
            if bare.lower() not in support:
                out.append(bare)

    # Preserve order, drop duplicates: the reason string quotes the first hit
    # and a repeated one adds nothing.
    seen: set[str] = set()
    unique: list[str] = []
    for item in out:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique


def _dig(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def add_banned_phrases(rubric: str, phrases: Iterable[str]) -> str:
    """Append phrases to the rubric's banned list, or add the bullet if absent.

    Used by the offline reflector. Kept here rather than in the reflector so the
    tag syntax has exactly one owner - a writer and a reader that disagree about
    the format is a bug that shows up as "tuning mysteriously does nothing".
    """
    new = [p.strip().lower() for p in phrases if p and p.strip()]
    if not new:
        return rubric

    existing = RuleEngine(rubric).banned_phrases()
    new = [p for p in dict.fromkeys(new) if p not in existing]
    if not new:
        return rubric

    match = re.search(r"\[banned:\s*(.*?)\]", rubric, re.DOTALL)
    if match:
        addition = ", ".join(f'"{p}"' for p in new)
        return rubric[: match.end() - 1] + ", " + addition + rubric[match.end() - 1 :]

    bullet = (
        "\n- Reject filler that would be equally true of any other item. "
        + "[banned: "
        + ", ".join(f'"{p}"' for p in new)
        + "]"
    )
    return rubric.rstrip() + bullet + "\n"
