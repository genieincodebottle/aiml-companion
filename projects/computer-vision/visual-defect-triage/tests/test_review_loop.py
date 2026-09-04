"""The loop's one silent failure, plus queue ordering and agreement."""
import pytest

from src.review.agreement import cohen_kappa
from src.review.capture import assert_not_in_evaluation
from src.review.queue import order, priority


def test_uncertainty_peaks_at_one_half():
    assert priority(0.5, 1.0) > priority(0.9, 1.0)
    assert priority(0.5, 1.0) > priority(0.1, 1.0)


def test_value_breaks_the_tie():
    assert priority(0.5, 5000.0) > priority(0.5, 5.0)


def test_queue_orders_by_expected_value():
    items = [
        {"id": "cheap_uncertain", "confidence": 0.5, "batch_value": 5.0},
        {"id": "costly_uncertain", "confidence": 0.5, "batch_value": 5000.0},
        {"id": "costly_certain", "confidence": 0.99, "batch_value": 5000.0},
    ]
    assert order(items)[0]["id"] == "costly_uncertain"


def test_mined_data_must_not_reach_the_evaluation_set():
    mined = [{"image_id": "img_1"}, {"image_id": "img_2"}]
    assert_not_in_evaluation(mined, {"img_9"})
    with pytest.raises(AssertionError):
        assert_not_in_evaluation(mined, {"img_2"})


def test_kappa_corrects_for_chance():
    """Two reviewers who both say pass 90 percent of the time agree by luck."""
    a = ["pass"] * 90 + ["scratch"] * 10
    b = ["pass"] * 90 + ["scratch"] * 10
    assert cohen_kappa(a, b) == pytest.approx(1.0)

    c = ["pass"] * 100
    d = ["pass"] * 90 + ["scratch"] * 10
    assert cohen_kappa(c, d) == pytest.approx(0.0, abs=1e-9)
