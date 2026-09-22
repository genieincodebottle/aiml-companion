"""The task the harness is pointed at, and the synthetic tickets it ships with.

The decision is one a support desk actually makes. A ticket arrives, and
something has to answer: does this go to a human specialist now, or does it join
the normal queue? Getting it wrong is expensive in both directions.

Every ticket here is generated, not collected. Its label is computed from the
five signals below, so a model that reads those five signals well will always
beat a model asked the whole question at once. That is true by construction and
it is not evidence about any real model. The generated set exists so the harness
can run end to end with no credentials. Point `--backend live` at your own
labelled tickets when you want an answer you can act on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# The five narrow signals the escalation decision decomposes into. The key is
# what the harness uses everywhere; the description is what a human labeller and
# the model both read.
SIGNALS: dict[str, str] = {
    "outage": "The customer says the service is completely unusable right now.",
    "deadline": "The customer names a deadline inside the next day.",
    "legal": "The customer raises a legal, contractual or compliance concern.",
    "repeat": "The customer says they have contacted support about this before.",
    "refund": "The customer explicitly asks for money back.",
}

# How much each signal moves the escalation decision. No single signal decides
# it, which is what makes the decision worth decomposing rather than reading off
# one keyword.
WEIGHTS: dict[str, float] = {
    "outage": 1.5,
    "deadline": 1.1,
    "legal": 1.3,
    "repeat": 0.9,
    "refund": 0.6,
}
BIAS = -2.2

# How decisive the desk is. Higher makes the label follow the signals more
# closely; lower buries them in noise. Real desks sit somewhere in between, and
# this is the knob to turn when you want to see what a noisier one does to the
# share you can automate.
SHARPNESS = 2.0

# How often each signal is present, and the sentence that carries it.
PREVALENCE: dict[str, float] = {
    "outage": 0.30,
    "deadline": 0.25,
    "legal": 0.18,
    "repeat": 0.35,
    "refund": 0.22,
}

PHRASES: dict[str, list[str]] = {
    "outage": [
        "Nothing loads at all, the whole dashboard is blank for every user on our team.",
        "The API has returned 503 for the last forty minutes and we are completely stuck.",
        "We cannot sign in. Every single person here is locked out.",
    ],
    "deadline": [
        "We go live tomorrow morning and this blocks the launch.",
        "Our board demo is in six hours.",
        "The migration window closes tonight.",
    ],
    "legal": [
        "Our legal team has asked me to record this as a contractual breach.",
        "This looks like a GDPR issue and our compliance officer is now involved.",
        "I need to know whether this counts against the uptime commitment in our contract.",
    ],
    "repeat": [
        "This is the third time I am writing about the same thing.",
        "I was told last week that someone would follow up. Nobody did.",
        "I have already been passed between two agents on this ticket.",
    ],
    "refund": [
        "I would like this month refunded.",
        "Please cancel the plan and return what we paid.",
        "At this point I want my money back.",
    ],
}

FILLER: list[str] = [
    "Thanks for reading this.",
    "Let me know what you need from my side.",
    "Happy to jump on a call if that is faster.",
    "I have attached a screenshot.",
    "Account is on the growth plan, if that matters.",
    "Apologies if this is the wrong channel.",
]

OPENERS: list[str] = [
    "Hi support,",
    "Hello,",
    "Hi team,",
    "Good morning,",
]


@dataclass(frozen=True)
class Ticket:
    """One labelled example. `signals` is the hidden truth a labeller would mark."""

    id: str
    text: str
    signals: dict[str, bool]
    escalate: bool


def _logistic(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    from math import exp

    return 1.0 / (1.0 + exp(-x))


def make_tickets(n: int = 600, seed: int = 7) -> list[Ticket]:
    """Generate `n` labelled tickets. Same seed, same tickets, on every machine."""
    rng = random.Random(seed)
    tickets: list[Ticket] = []

    for i in range(n):
        signals = {name: rng.random() < p for name, p in PREVALENCE.items()}

        parts = [rng.choice(OPENERS)]
        # Shuffle the evidence sentences so position never encodes the label.
        present = [name for name, on in signals.items() if on]
        rng.shuffle(present)
        for name in present:
            parts.append(rng.choice(PHRASES[name]))
        for _ in range(rng.randint(1, 2)):
            parts.append(rng.choice(FILLER))

        score = SHARPNESS * (BIAS + sum(WEIGHTS[name] for name, on in signals.items() if on))
        # Label noise, so nothing can reach 100% and the calibration numbers stay honest.
        escalate = rng.random() < _logistic(score)

        tickets.append(
            Ticket(
                id=f"T{i:04d}",
                text=" ".join(parts),
                signals=signals,
                escalate=escalate,
            )
        )

    return tickets


def split(tickets: list[Ticket], fraction: float = 0.5) -> tuple[list[Ticket], list[Ticket]]:
    """Fit half, score the other half. Never report a number from the fitting half."""
    cut = int(len(tickets) * fraction)
    return tickets[:cut], tickets[cut:]
