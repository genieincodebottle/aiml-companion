"""The comparison the harness exists to run.

Both arms see the same tickets and the same split. Both get one thing fitted on
the training half, so neither is handed an unfair advantage.

    single       one holistic question. A temperature is fitted on the training
                 half, which is the only knob a single probability gives you.
    decomposed   five narrow questions. A logistic regression is fitted on the
                 training half, then a temperature on top of it.

Everything printed comes from the held-out half.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import Backend
from .calibrate import (
    LogisticModel,
    OperatingPoint,
    Platt,
    accuracy,
    apply_temperature,
    ece,
    fit_logistic,
    fit_platt,
    fit_temperature,
    operating_point,
)
from .questions import DECOMPOSED, SINGLE
from .task import SIGNALS, Ticket, split


def state_of(ticket: Ticket) -> dict:
    """What the model is shown. The hidden signals and the label never appear here."""
    return {"id": ticket.id, "ticket": ticket.text}


def collect(backend: Backend, tickets: list[Ticket], question_set) -> list[dict[str, float]]:
    """One call per ticket. Every question in the set is answered in that one call."""
    rows = []
    for ticket in tickets:
        answers = backend.ask(state_of(ticket), question_set.questions)
        rows.append({key: float(a.noul) for key, a in answers.items()})
    return rows


@dataclass
class ArmResult:
    name: str
    accuracy: float
    ece_raw: float
    ece_temperature: float
    ece_platt: float
    temperature: float
    platt: Platt
    calls: int
    questions_per_call: int
    point: OperatingPoint | None
    model: LogisticModel | None = None
    probs: list[float] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)


def _finish(
    name: str,
    train_p: list[float],
    train_labels: list[bool],
    test_p: list[float],
    labels: list[bool],
    *,
    calls: int,
    questions_per_call: int,
    target: float,
    model: LogisticModel | None = None,
) -> ArmResult:
    """Fit both calibrators on the training half, report everything on the held-out half."""
    temperature = fit_temperature(train_p, train_labels)
    platt = fit_platt(train_p, train_labels)

    by_temperature = apply_temperature(test_p, temperature)
    by_platt = platt.apply(test_p)

    return ArmResult(
        name=name,
        accuracy=accuracy(by_platt, labels),
        ece_raw=ece(test_p, labels),
        ece_temperature=ece(by_temperature, labels),
        ece_platt=ece(by_platt, labels),
        temperature=temperature,
        platt=platt,
        calls=calls,
        questions_per_call=questions_per_call,
        point=operating_point(by_platt, labels, target),
        model=model,
        probs=by_platt,
        labels=labels,
    )


def run_single(backend: Backend, train: list[Ticket], test: list[Ticket], target: float) -> ArmResult:
    train_p = [r["escalate"] for r in collect(backend, train, SINGLE)]
    test_p = [r["escalate"] for r in collect(backend, test, SINGLE)]

    return _finish(
        "single",
        train_p,
        [t.escalate for t in train],
        test_p,
        [t.escalate for t in test],
        calls=len(train) + len(test),
        questions_per_call=1,
        target=target,
    )


def run_decomposed(backend: Backend, train: list[Ticket], test: list[Ticket], target: float) -> ArmResult:
    features = list(SIGNALS)
    train_rows = [[r[f] for f in features] for r in collect(backend, train, DECOMPOSED)]
    test_rows = [[r[f] for f in features] for r in collect(backend, test, DECOMPOSED)]
    train_labels = [t.escalate for t in train]
    labels = [t.escalate for t in test]

    model = fit_logistic(train_rows, train_labels, features)

    return _finish(
        "decomposed",
        [model.predict(r) for r in train_rows],
        train_labels,
        [model.predict(r) for r in test_rows],
        labels,
        calls=len(train) + len(test),
        questions_per_call=len(features),
        target=target,
        model=model,
    )


def compare(backend: Backend, tickets: list[Ticket], target: float = 0.95) -> list[ArmResult]:
    train, test = split(tickets)
    return [
        run_single(backend, train, test, target),
        run_decomposed(backend, train, test, target),
    ]
