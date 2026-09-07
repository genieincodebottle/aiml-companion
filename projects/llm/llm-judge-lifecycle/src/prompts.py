"""Every prompt the system sends, in one file.

Prompts are the interface between four roles and three vendors, and scattering
them across the modules that happen to call them is how a judge prompt and a
rater guideline quietly drift apart. Keeping them here means a reviewer can
read the entire thing the model sees without opening a single other file.

Two conventions hold throughout:

**Fenced payloads.** Structured data goes inside ``=== NAME (JSON) ===``
fences (see ``providers/_common.py``). It is ordinary prompt hygiene - a model
guessing where the data stops is a model that sometimes guesses wrong - and it
is what lets the offline stub parse real inputs instead of returning canned
text.

**The rubric is passed in, never baked in.** Every judge call takes its rubric
as a parameter. That is what makes Phase II possible: RART changes the rubric
between iterations and nothing else moves, so any change in the metrics is
attributable to the rubric and to nothing else. A prompt with the criteria
hardcoded cannot be tuned, only rewritten.
"""

from __future__ import annotations

from typing import Any

from .domain import Criterion, Domain, Record
from .providers._common import block, text_block

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a quality judge. You apply ONE written criterion to ONE short artefact \
and return a verdict.

You are not an editor and not a critic. Do not suggest improvements, do not \
comment on style, and do not apply any standard that is not written in the \
rubric you are given. If the rubric does not forbid it, it passes.

Work in this order: state your reason, then the verdict that follows from it, \
then the failure mode if you failed it. Reaching for the verdict first and \
explaining afterwards produces a confident label and a reason invented to match \
it.

The reason is not decoration. When you reject something, your reason is handed \
to the writer as their instruction for the next attempt, so a vague reason \
produces a vague revision.

Most artefacts you see will pass. Say so when they do.\
"""

# The judge returns a reason AND a label AND a mode. The reason is required for a
# FAIL because Phase III feeds it back to the generator, and the mode is required
# because Phase II scores whether the judge failed things for the same reason the
# human did. A judge that returns only a label cannot be reasoning-aligned,
# cannot drive a revision loop, and cannot be debugged.
#
# `reason` COMES FIRST, and the field order is load-bearing rather than
# stylistic. Structured decoding emits fields in schema order, so with `label`
# first the model commits to a verdict token before doing any of the work, and
# the reason that follows can only rationalise it.
#
# That is not a theory. Measured here, the judge said it out loud:
#
#     "The explanation is 29 words long, but the prompt requested a strict
#      evaluation against the 40-word limit which this actually passes.
#      However, following the exact rubric instructions..."      -> FAIL
#
# Twenty-nine words against a forty-word limit: correctly counted, correctly
# described as passing, and failed anyway. Putting `reason` first makes the model
# do the work before it picks a side - chain-of-thought bought with field order -
# and it matters most exactly where thinking is disabled, which for this role it
# is.
#
# Measured over ten in-limit artefacts: label-first got 3 wrong, reason-first got
# 0. See docs/results.md.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "failure_mode": {"type": "string"},
        "label": {"type": "string", "enum": ["PASS", "FAIL"]},
    },
    "required": ["reason", "label"],
}


def judge_prompt(
    domain: Domain, criterion: Criterion, rubric: str, record: Record, artefact: str
) -> str:
    modes = ", ".join(criterion.failure_modes) or "none declared"
    return "\n\n".join(
        [
            f"Judge one {domain.artefact_noun} against the criterion "
            f"{criterion.display!r}.",
            text_block("RUBRIC", rubric),
            block("RECORD", record.as_dict()),
            text_block("ARTEFACT", artefact),
            # The closing instruction, and every clause in it is load-bearing.
            #
            # The earlier version ended "...the failure mode that best fits,
            # chosen from that list", which reads as an instruction to FIND a
            # failure rather than as a vocabulary for describing one. Measured,
            # it produced a systematic false-FAIL bias - and on a single-mode
            # criterion the judge said so out loud: "the system has flagged it
            # as a failure to demonstrate the evaluation process for the
            # 'too_long' failure mode."
            #
            # Rewording this alone took six in-limit artefacts from 5 wrong to
            # 2. Reordering JUDGE_SCHEMA so the reason precedes the label took
            # the remainder to 0. Neither fix was sufficient alone.
            "Decide PASS or FAIL on this criterion alone.\n"
            "State your reason FIRST, then the verdict that follows from it.\n"
            "If, and only if, the verdict is FAIL, name the failure mode from "
            f"this list: {modes}. The list describes how failures are "
            "categorised. It is not a suggestion that this artefact fails.",
        ]
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generator_prompt(
    domain: Domain, record: Record, *, critique: str | None = None, attempt: int = 0
) -> str:
    parts = [
        domain.generation.get("instruction", "").strip(),
        block("RECORD", record.as_dict()),
        block("ATTEMPT", attempt),
    ]
    if critique:
        # The critic role. The judge's rejection reason becomes the writer's
        # instruction, which is the mechanism that makes revision better than
        # resampling - and also why a right-verdict-but-wrong-reason rejection
        # is a real defect rather than a philosophical one. The wrong reason is
        # not merely unhelpful here; it actively steers the next draft towards
        # fixing something that was never broken.
        parts.append(
            text_block("CRITIQUE", critique)
            + "\n\nYour previous attempt was rejected for the reason above. "
            "Fix precisely that. Do not rewrite the parts that were not "
            "criticised - they passed."
        )
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Reasoning meta-judge
# ---------------------------------------------------------------------------

META_JUDGE_SYSTEM = """\
You compare two explanations of why the same artefact failed: one written by a \
human rater, one written by an automated judge.

You are NOT deciding whether either is correct. Both already agree the artefact \
fails. You are deciding whether they are talking about the SAME defect.

Answer RATIONALE_AGREEMENT when they identify the same underlying problem, even \
in different words. Answer RATIONALE_MISMATCH when the judge has landed on the \
right verdict for a different reason than the human's.

When it is genuinely ambiguous, answer RATIONALE_AGREEMENT. A false mismatch \
sends rubric tuning off to fix something that was never broken, which is more \
expensive than missing one real disagreement.\
"""

META_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["RATIONALE_AGREEMENT", "RATIONALE_MISMATCH"],
        },
        "note": {"type": "string"},
    },
    "required": ["verdict"],
}


def meta_judge_prompt(
    artefact: str,
    judge_reason: str,
    judge_mode: str | None,
    human_rationale: str,
    human_mode: str | None,
) -> str:
    return "\n\n".join(
        [
            "Do these two accounts identify the same defect?",
            text_block("ARTEFACT", artefact),
            text_block("JUDGE_REASON", judge_reason),
            text_block("JUDGE_FAILURE_MODE", judge_mode or ""),
            text_block("HUMAN_RATIONALE", human_rationale),
            text_block("HUMAN_FAILURE_MODE", human_mode or ""),
        ]
    )


# ---------------------------------------------------------------------------
# Reflector (the RART optimiser)
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM = """\
You improve a written rubric so that an automated judge applying it agrees more \
often with human raters.

You are given the current rubric and a FOCUS SET: the examples where the judge \
went wrong. Two kinds appear there, and they need different fixes.

  label mismatch  - the judge and the human disagreed on PASS versus FAIL.
  reason mismatch - they agreed it fails, but for different reasons. The rubric
                    is catching the right cases by accident, so it will stop
                    catching them as soon as the inputs shift.

Rules for the rubric you return:

1. Return the COMPLETE revised rubric, not a diff and not a commentary.
2. Keep every clause that is working. You are editing, not rewriting. A rubric
   that changes wholesale between iterations cannot be attributed to anything.
3. Do not encode individual examples. A clause that only recognises the exact
   sentences in the focus set will score well on them and worse on everything
   else. Generalise the pattern or leave it alone.
4. Preserve any bracketed tags such as [banned: ...] or [grounded]. They are
   machine-readable and dropping one silently disables a check.
5. Prefer one sharper clause to three vague ones. Rubrics that grow every
   iteration end up self-contradictory, and a contradictory rubric produces a
   judge whose verdict depends on which clause it happened to weigh.\
"""


def reflector_prompt(
    criterion: Criterion, rubric: str, focus: list[dict[str, Any]]
) -> str:
    return "\n\n".join(
        [
            f"Revise the rubric for the criterion {criterion.display!r}.",
            text_block("RUBRIC", rubric),
            block("FOCUS", focus),
            "Return only the revised rubric.",
        ]
    )


# ---------------------------------------------------------------------------
# Benchmark synthesis (Phase I, source ii)
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You write BOUNDARY cases for an evaluation benchmark.

A boundary case sits as close to the pass/fail line as you can make it while \
still falling clearly on the side you were asked for. Obvious examples are \
useless here - naturalistic sampling already produces those in quantity, and a \
judge that only ever sees easy cases looks excellent and is untested.

Write the artefact only. Do not label it, do not explain it, and do not hedge \
it. A human decides the label.\
"""


def synthesis_prompt(
    domain: Domain, criterion: Criterion, record: Record, target: str
) -> str:
    return "\n\n".join(
        [
            f"Write one {domain.artefact_noun} that a careful human rater would "
            f"label {target} on the criterion {criterion.display!r}, and that a "
            "careless one might well get wrong.",
            text_block("RUBRIC", criterion.guideline),
            block("RECORD", record.as_dict()),
            domain.generation.get("instruction", "").strip(),
        ]
    )
