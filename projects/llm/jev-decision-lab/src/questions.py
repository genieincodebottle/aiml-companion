"""The two ways to ask the same question, plus one example of each primitive.

A typed decision model takes some state and a set of questions, answers them in
parallel and in isolation against that state, and returns typed values. The
questions are the part you design. This module holds them in a backend-neutral
form so the simulated and live backends read the same definitions.

Three primitives, from the TypeSafe System One API:

    Noul      yes or no, returned as a probability between 0 and 1
    Choice    one option from a set, up to 255 of them, with a distribution
    Score     a position on an ordered scale of 2 to 10 levels, returned as the
              probability-weighted mean of the level numbers, so it is a decimal
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .task import SIGNALS


@dataclass(frozen=True)
class NoulQ:
    instructions: str


@dataclass(frozen=True)
class ChoiceQ:
    instructions: str
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class ScoreQ:
    instructions: str
    criteria: list[str]


Question = NoulQ | ChoiceQ | ScoreQ


@dataclass(frozen=True)
class QuestionSet:
    """A named set of questions asked against one ticket."""

    name: str
    questions: dict[str, Question] = field(default_factory=dict)


# Ask the whole decision at once. This is the shape most people reach for first,
# and the one that leaves the model to weigh evidence you never see.
SINGLE = QuestionSet(
    name="single",
    questions={
        "escalate": NoulQ(
            instructions=(
                "This support ticket needs a human specialist now, rather than "
                "joining the normal queue."
            )
        ),
    },
)

# Ask five narrow questions instead, each one answerable from the text without
# judgement about what the answer implies. What the signals add up to is decided
# afterwards, by a regression fitted on your own labels.
DECOMPOSED = QuestionSet(
    name="decomposed",
    questions={name: NoulQ(instructions=text) for name, text in SIGNALS.items()},
)

# Not used by the comparison. It is here so the file shows all three primitives
# in the shape the SDK expects them.
PRIMITIVE_DEMO = QuestionSet(
    name="primitives",
    questions={
        "outage": NoulQ(instructions=SIGNALS["outage"]),
        "queue": ChoiceQ(
            instructions="Which queue should handle this ticket",
            criteria={
                "billing": "Payments, invoices, refunds and plan changes.",
                "technical": "Errors, outages and integration problems.",
                "account": "Access, permissions and user management.",
            },
        ),
        "frustration": ScoreQ(
            instructions="How frustrated the customer sounds",
            criteria=[
                "Calm, just reporting facts",
                "Mildly annoyed",
                "Clearly frustrated but civil",
                "Angry, threatening to leave",
            ],
        ),
    },
)

SETS = {s.name: s for s in (SINGLE, DECOMPOSED, PRIMITIVE_DEMO)}


def to_sdk(questions: dict[str, Question]) -> dict:
    """Translate to typesafe_sdk objects. Imported lazily so the offline path needs no install."""
    from typesafe_sdk import Choice, Noul, Score  # noqa: PLC0415

    out = {}
    for key, q in questions.items():
        if isinstance(q, NoulQ):
            out[key] = Noul(instructions=q.instructions)
        elif isinstance(q, ChoiceQ):
            out[key] = Choice(instructions=q.instructions, criteria=q.criteria)
        elif isinstance(q, ScoreQ):
            out[key] = Score(instructions=q.instructions, criteria=q.criteria)
        else:
            raise TypeError(f"unknown question type for {key!r}: {type(q).__name__}")
    return out
