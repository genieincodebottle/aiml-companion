"""The maths, checked against cases where the right answer is known in advance."""

from __future__ import annotations

import random

from src.calibrate import (
    accuracy,
    apply_temperature,
    ece,
    fit_logistic,
    fit_platt,
    fit_temperature,
    logistic,
    logit,
    operating_point,
    reliability,
)


def test_logit_and_logistic_are_inverses():
    for p in (0.01, 0.2, 0.5, 0.75, 0.999):
        assert abs(logistic(logit(p)) - p) < 1e-9


def test_logistic_saturates_without_overflow():
    assert logistic(-1000) == 0.0
    assert logistic(1000) == 1.0


def test_fit_logistic_recovers_known_weights():
    rng = random.Random(1)
    rows, labels = [], []
    for _ in range(1500):
        a, b = rng.random(), rng.random()
        rows.append([a, b])
        labels.append(rng.random() < logistic(4.0 * a - 2.0 * b - 1.0))

    model = fit_logistic(rows, labels, ["a", "b"], l2=0.0, iterations=6000, learning_rate=0.5)
    assert model.weights[0] > 2.5          # positive and clearly the stronger one
    assert model.weights[1] < -1.0         # negative
    assert model.weights[0] > abs(model.weights[1])


def test_perfect_probabilities_have_no_calibration_error():
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = [False, False, True, True]
    assert ece(probs, labels) == 0.0
    assert accuracy(probs, labels) == 1.0


def test_ece_catches_a_model_that_is_always_too_sure():
    probs = [0.9] * 100
    labels = [i < 50 for i in range(100)]   # says 90%, happens 50% of the time
    assert abs(ece(probs, labels) - 0.4) < 1e-6


def test_reliability_bins_cover_every_row():
    rng = random.Random(2)
    probs = [rng.random() for _ in range(300)]
    labels = [rng.random() < p for p in probs]
    assert sum(b.count for b in reliability(probs, labels)) == 300


def test_temperature_cools_a_model_that_is_too_sharp():
    rng = random.Random(3)
    truth = [rng.random() for _ in range(2000)]
    labels = [rng.random() < p for p in truth]
    too_sharp = [logistic(logit(p) * 2.5) for p in truth]

    t = fit_temperature(too_sharp, labels)
    assert t > 1.5
    assert ece(apply_temperature(too_sharp, t), labels) < ece(too_sharp, labels)


def test_temperature_cannot_fix_a_shift_but_platt_can():
    rng = random.Random(4)
    truth = [rng.random() for _ in range(2000)]
    labels = [rng.random() < p for p in truth]
    shifted = [logistic(logit(p) + 1.4) for p in truth]   # leans yes on every row

    by_temperature = apply_temperature(shifted, fit_temperature(shifted, labels))
    by_platt = fit_platt(shifted, labels).apply(shifted)

    assert ece(by_platt, labels) < ece(by_temperature, labels)
    assert ece(by_platt, labels) < 0.05


def test_operating_point_trades_coverage_for_accuracy():
    rng = random.Random(5)
    probs = [rng.random() for _ in range(500)]
    labels = [rng.random() < p for p in probs]

    loose = operating_point(probs, labels, 0.70)
    strict = operating_point(probs, labels, 0.90)
    assert loose is not None
    assert strict is None or strict.coverage <= loose.coverage
    assert loose.accuracy_in_band >= 0.70


def test_operating_point_returns_nothing_when_the_bar_is_impossible():
    probs = [0.5] * 50
    labels = [i % 2 == 0 for i in range(50)]
    assert operating_point(probs, labels, 0.99) is None
