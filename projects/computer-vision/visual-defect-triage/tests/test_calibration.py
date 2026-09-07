"""Temperature scaling is safe to deploy because it cannot reorder logits."""
import numpy as np

from src.calibrate import fit_temperature, softmax
from src.metrics.calibration import expected_calibration_error


def test_temperature_never_changes_the_prediction():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, size=(500, 7))
    labels = logits.argmax(1)

    t = fit_temperature(logits, labels)
    assert np.array_equal(logits.argmax(1), (logits / t).argmax(1)), "temperature changed a prediction"


def test_temperature_softens_overconfident_logits():
    """Overconfident logits are large in magnitude, so the fit should exceed 1."""
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 6, size=(800, 7))
    labels = rng.integers(0, 7, size=800)
    assert fit_temperature(logits, labels) > 1.0


def test_calibration_improves_ece():
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 5, size=(1500, 7))
    labels = np.array([rng.choice(7, p=softmax(row, 5.0)) for row in logits])

    t = fit_temperature(logits, labels)
    pred = logits.argmax(1)
    correct = pred == labels

    raw = softmax(logits, 1.0)[np.arange(len(pred)), pred]
    cal = softmax(logits, t)[np.arange(len(pred)), pred]
    assert expected_calibration_error(cal, correct) < expected_calibration_error(raw, correct)


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.default_rng(3).normal(size=(20, 7)), 1.8)
    assert np.allclose(p.sum(axis=1), 1.0)
