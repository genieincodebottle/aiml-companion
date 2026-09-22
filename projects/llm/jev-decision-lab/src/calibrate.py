"""Turning probabilities into a decision you can defend.

Four things happen here, and none of them depend on which model produced the
probabilities. That is the point. Point this at any decision model, with your own
labels, and it answers the only question that matters before you ship one.

    fit_logistic       learn what the signals are worth on this desk
    ece                how far the confidence numbers are from the truth
    fit_temperature    the one-parameter fix for confidence that runs hot or cold
    fit_platt          the two-parameter fix, for confidence that is also shifted
    operating_point    the confidence band where accuracy clears your bar, and
                       how much traffic that band covers

Everything is fitted on one half of the labels and reported on the other. A
number reported on the half it was fitted on is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

EPS = 1e-9


def logistic(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + exp(-x))


def logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return log(p / (1 - p))


# ── Fitting what the signals are worth ────────────────────────────────────────

@dataclass
class LogisticModel:
    weights: list[float]
    bias: float
    features: list[str]

    def predict(self, row: list[float]) -> float:
        return logistic(self.bias + sum(w * x for w, x in zip(self.weights, row)))


def fit_logistic(
    rows: list[list[float]],
    labels: list[bool],
    features: list[str],
    *,
    l2: float = 1.0,
    iterations: int = 4000,
    learning_rate: float = 0.3,
) -> LogisticModel:
    """Plain batch gradient descent. Five features and a few hundred rows do not need more."""
    if not rows:
        raise ValueError("no rows to fit")
    n_features = len(rows[0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(rows)

    for _ in range(iterations):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for row, label in zip(rows, labels):
            error = logistic(bias + sum(w * x for w, x in zip(weights, row))) - (1.0 if label else 0.0)
            grad_b += error
            for j, x in enumerate(row):
                grad_w[j] += error * x
        bias -= learning_rate * grad_b / n
        for j in range(n_features):
            weights[j] -= learning_rate * (grad_w[j] / n + l2 * weights[j] / n)

    return LogisticModel(weights=weights, bias=bias, features=features)


# ── How good, and how honest ──────────────────────────────────────────────────

def accuracy(probs: list[float], labels: list[bool], threshold: float = 0.5) -> float:
    if not probs:
        return 0.0
    hits = sum(1 for p, y in zip(probs, labels) if (p >= threshold) == y)
    return hits / len(probs)


@dataclass(frozen=True)
class Bin:
    low: float
    high: float
    count: int
    mean_confidence: float
    observed_rate: float


def reliability(probs: list[float], labels: list[bool], bins: int = 10) -> list[Bin]:
    """Bucket by predicted probability, then compare what was promised with what happened."""
    out: list[Bin] = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        chosen = [
            (p, y)
            for p, y in zip(probs, labels)
            if (p >= low and p < high) or (i == bins - 1 and p == 1.0)
        ]
        if not chosen:
            out.append(Bin(low, high, 0, 0.0, 0.0))
            continue
        mean_p = sum(p for p, _ in chosen) / len(chosen)
        rate = sum(1 for _, y in chosen if y) / len(chosen)
        out.append(Bin(low, high, len(chosen), mean_p, rate))
    return out


def ece(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    """Expected calibration error. 0.02 says a 0.80 really happens about 80% of the time."""
    if not probs:
        return 0.0
    total = len(probs)
    return sum(
        b.count / total * abs(b.mean_confidence - b.observed_rate)
        for b in reliability(probs, labels, bins)
        if b.count
    )


def fit_temperature(probs: list[float], labels: list[bool]) -> float:
    """The one-parameter fix. Above 1 cools confidence down, below 1 warms it up."""
    best_t, best_loss = 1.0, float("inf")
    t = 0.05
    while t <= 6.0:
        loss = 0.0
        for p, y in zip(probs, labels):
            q = min(max(logistic(logit(p) / t), EPS), 1 - EPS)
            loss -= log(q) if y else log(1 - q)
        if loss < best_loss:
            best_loss, best_t = loss, t
        t += 0.05
    return round(best_t, 2)


def apply_temperature(probs: list[float], t: float) -> list[float]:
    return [logistic(logit(p) / t) for p in probs]


@dataclass(frozen=True)
class Platt:
    """logistic(slope * logit(p) + shift). Temperature is this with the shift nailed to zero."""

    slope: float
    shift: float

    def apply(self, probs: list[float]) -> list[float]:
        return [logistic(self.slope * logit(p) + self.shift) for p in probs]


def fit_platt(probs: list[float], labels: list[bool]) -> Platt:
    """Fit a slope and a shift together.

    Temperature can only make a probability sharper or flatter. It pulls
    everything towards 0.5 and leaves the middle of the distribution where it
    was. So it fixes a model that is uniformly too sure of itself, and it cannot
    touch a model that is sure in the wrong direction, which is what you get when
    the model weighed your evidence differently from the way you do. That needs
    the shift as well, and it costs one more parameter.
    """
    model = fit_logistic([[logit(p)] for p in probs], labels, ["logit"], l2=0.0, iterations=3000)
    return Platt(slope=model.weights[0], shift=model.bias)


# ── The band you actually ship ────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatingPoint:
    confidence: float
    coverage: float
    accuracy_in_band: float
    handled: int
    total: int


def operating_point(
    probs: list[float],
    labels: list[bool],
    target_accuracy: float = 0.95,
    *,
    min_coverage: float = 0.05,
) -> OperatingPoint | None:
    """The lowest confidence cut whose automated slice still clears `target_accuracy`.

    Confidence is how far a probability sits from the fence, so 0.03 and 0.97 are
    both confident. Everything below the cut goes to a human.

    `min_coverage` is not decoration. Walk the sorted list far enough and the top
    handful of rows will always look perfect, so an unguarded search reports a
    band of three tickets at 100% and someone quotes it. A band that automates
    less than this share is not worth running, so it is reported as no band.
    """
    scored = sorted(
        ((max(p, 1 - p), p, y) for p, y in zip(probs, labels)), key=lambda r: -r[0]
    )
    total = len(scored)
    floor = max(1, int(total * min_coverage))
    best: OperatingPoint | None = None

    for i in range(floor, total + 1):
        band = scored[:i]
        hits = sum(1 for _, p, y in band if (p >= 0.5) == y)
        acc = hits / i
        if acc >= target_accuracy:
            best = OperatingPoint(
                confidence=round(band[-1][0], 4),
                coverage=i / total,
                accuracy_in_band=acc,
                handled=i,
                total=total,
            )
    return best
