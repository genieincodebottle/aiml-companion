"""A synthetic support inbox that violates the Naive Bayes assumption on purpose.

Naive Bayes assumes that, once you know the class, the tokens are independent:

    P(token_a, token_b | class) == P(token_a | class) * P(token_b | class)

On real text that is obviously false. "credit" and "card" do not appear
independently in a billing ticket. The usual response is to shrug and say the
model works anyway, which is true and unsatisfying, because nobody measures how
false the assumption is or what the falseness actually costs.

Here it is false BY CONSTRUCTION and by a known amount. `DEPENDENT_PAIRS` lists
the token pairs that are wired to fire together, and `dependency_strength` says
how hard. At strength 0 the assumption holds exactly and Naive Bayes is the
correct model for this data. At 0.95 the pairs are nearly welded together and
the model counts the same evidence twice.

That turns "it works anyway" into a measurable claim you can plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------- the answer key
CLASSES = ("billing", "login_access", "bug_crash",
           "feature_request", "shipping", "abuse_report")

# Deliberately imbalanced, the way a real inbox is. abuse_report is 3%, which is
# where accuracy quietly stops being a useful metric.
CLASS_PRIORS = {
    "billing": 0.32, "login_access": 0.24, "bug_crash": 0.19,
    "feature_request": 0.12, "shipping": 0.10, "abuse_report": 0.03,
}

# Tokens that genuinely carry class signal, with the per-class emission
# probability the generator uses. This IS the ground truth a model tries to
# recover, so nothing in the fitting path may read it.
SIGNAL_TOKENS: dict[str, dict[str, float]] = {
    "billing": {"invoice": 0.55, "charged": 0.50, "card": 0.45, "refund": 0.40,
                "subscription": 0.35, "payment": 0.30, "vat": 0.12,
                "billed": 0.30, "debited": 0.26, "statement": 0.22},
    "login_access": {"password": 0.58, "reset": 0.45, "locked": 0.40,
                     "otp": 0.35, "signin": 0.33, "account": 0.28, "2fa": 0.15,
                     "credentials": 0.28, "login": 0.30, "lockout": 0.20},
    "bug_crash": {"crash": 0.55, "error": 0.48, "stacktrace": 0.32,
                  "freeze": 0.30, "reproduce": 0.28, "version": 0.25,
                  "exception": 0.20, "traceback": 0.24, "fatal": 0.20,
                  "panic": 0.16},
    "feature_request": {"suggest": 0.50, "wishlist": 0.35, "improve": 0.33,
                        "roadmap": 0.28, "export": 0.25, "integration": 0.22,
                        "enhancement": 0.24, "request": 0.26},
    "shipping": {"delivery": 0.55, "courier": 0.42, "tracking": 0.40,
                 "parcel": 0.35, "delayed": 0.30, "address": 0.25,
                 "dispatch": 0.24, "shipment": 0.26},
    "abuse_report": {"harassment": 0.50, "abusive": 0.45, "report": 0.40,
                     "block": 0.32, "threat": 0.25, "spam": 0.20,
                     "misconduct": 0.22, "offensive": 0.26},
}

# The planted violations, as CLUSTERS rather than pairs.
#
# A pair is too gentle to show the real failure. In actual text a billing
# ticket does not contain one redundant word, it contains a whole family of
# them: card, charged, billed, debited, statement all say the same thing. Naive
# Bayes multiplies a likelihood ratio for every one of them, so a cluster of
# size k counts the same evidence k times and the log-odds grow linearly in k.
# That is where the textbook overconfidence actually comes from.
#
# Each entry is (class, anchor, redundant partners). At `dependency_strength`
# s, every partner follows the anchor with probability s.
DEPENDENT_CLUSTERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("billing", "card", ("charged", "billed", "debited", "statement", "payment")),
    ("login_access", "password",
     ("reset", "credentials", "login", "lockout", "locked")),
    ("bug_crash", "crash", ("stacktrace", "traceback", "exception", "fatal", "panic")),
    ("feature_request", "suggest", ("wishlist", "enhancement", "request", "improve")),
    ("shipping", "delivery", ("tracking", "dispatch", "shipment", "courier")),
    ("abuse_report", "harassment", ("abusive", "offensive", "misconduct", "threat")),
)

#: The same information flattened to pairs, which is the shape the dependence
#: report wants. Every partner is paired with its anchor.
DEPENDENT_PAIRS: tuple[tuple[str, str, str], ...] = tuple(
    (cls, anchor, partner)
    for cls, anchor, partners in DEPENDENT_CLUSTERS
    for partner in partners
)

# Politeness that appears in every class at the same rate. It carries zero
# information, and a model that leans on it has learned nothing.
BOILERPLATE = ("hi", "hello", "thanks", "please", "regards", "urgent",
               "team", "asap", "kindly")

VOCAB: tuple[str, ...] = tuple(sorted(
    {t for toks in SIGNAL_TOKENS.values() for t in toks} | set(BOILERPLATE)))


def _emit_class(rng: np.random.Generator, cls: str, strength: float,
                boilerplate_rate: float) -> list[str]:
    """One ticket's bag of tokens, with the planted couplings applied."""
    emitted: set[str] = set()
    probs = SIGNAL_TOKENS[cls]

    # 1. independent draws, exactly what Naive Bayes assumes is the whole story
    for token, p in probs.items():
        if rng.random() < p:
            emitted.add(token)

    # 2. the couplings. When the anchor fired, drag its partner in with
    #    probability `strength`; when it did not, suppress the partner by the
    #    same amount. Both directions matter: dependence is not only about
    #    co-presence, it is also about co-absence.
    for pair_cls, anchor, partners in DEPENDENT_CLUSTERS:
        if pair_cls != cls:
            continue
        for partner in partners:
            if anchor in emitted:
                if rng.random() < strength:
                    emitted.add(partner)
            elif partner in emitted and rng.random() < strength:
                emitted.discard(partner)

    # 3. boilerplate, identical across classes
    for token in BOILERPLATE:
        if rng.random() < boilerplate_rate / len(BOILERPLATE) * 3:
            emitted.add(token)

    # 4. a little cross-class leakage, because real tickets are messy and a
    #    model that never sees an off-class token is being flattered
    for other, toks in SIGNAL_TOKENS.items():
        if other == cls:
            continue
        for token in toks:
            if rng.random() < 0.012:
                emitted.add(token)

    return sorted(emitted)


def generate(n_tickets: int = 9000, seed: int = 42,
             dependency_strength: float = 0.85,
             boilerplate_rate: float = 0.55) -> pd.DataFrame:
    """Build the inbox. One row per ticket, one column per vocabulary token.

    Returns a binary bag-of-words matrix plus `category` (the label) and
    `text` (the tokens joined, so the notebook can show something readable).
    """
    if n_tickets < len(CLASSES) * 20:
        raise ValueError(
            f"n_tickets={n_tickets} is too small to contain the 3% class at a "
            f"usable size; use at least {len(CLASSES) * 20}")

    rng = np.random.default_rng(seed)
    names = list(CLASS_PRIORS)
    weights = np.array([CLASS_PRIORS[c] for c in names], dtype=float)
    weights = weights / weights.sum()
    labels = rng.choice(names, size=n_tickets, p=weights)

    rows = np.zeros((n_tickets, len(VOCAB)), dtype=np.int8)
    index = {t: i for i, t in enumerate(VOCAB)}
    texts = []
    for r, cls in enumerate(labels):
        tokens = _emit_class(rng, cls, dependency_strength, boilerplate_rate)
        for t in tokens:
            rows[r, index[t]] = 1
        texts.append(" ".join(tokens))

    df = pd.DataFrame(rows, columns=list(VOCAB))
    df.insert(0, "ticket_id", [f"T{i:06d}" for i in range(n_tickets)])
    df["category"] = labels
    df["text"] = texts
    df["n_tokens"] = rows.sum(axis=1)
    return df


def token_columns(df: pd.DataFrame) -> list[str]:
    """The model-facing columns: the vocabulary, and nothing else."""
    return [c for c in df.columns if c in set(VOCAB)]
