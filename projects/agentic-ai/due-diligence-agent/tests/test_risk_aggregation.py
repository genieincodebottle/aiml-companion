"""
Regression tests for risk aggregation.

The bug
-------
`calculate_risk_score` divided the summed severity x likelihood by
`16 * len(risks)` -- a plain average. That makes the score go DOWN as the risk
register grows, so a thorough analyst produces a calmer report than a lazy one.
Measured on the old implementation:

    1 critical / very-likely risk, alone   -> 1.000  "critical"
    the same risk + 3 trivial ones         -> 0.273  "moderate"
    the same risk + 10 trivial ones        -> 0.119  "low"
    the same risk + 30 trivial ones        -> 0.062  "low"

The company did not get safer across those rows. Worse, the dilution is
trivially gameable: pad the list with rare/low items and any finding can be
argued down to "low".

The fix scores `max(peak, breadth)` -- the worst single item can never be
diluted, while a long tail of moderate items still escalates. The old
implementation fails every dilution test below.

Run: pytest tests/test_risk_aggregation.py -v
"""
import pytest

from src.tools.calculators import calculate_risk_score

CRITICAL = {"severity": "critical", "likelihood": "very_likely"}
HIGH = {"severity": "high", "likelihood": "likely"}
MEDIUM = {"severity": "medium", "likelihood": "possible"}
TRIVIAL = {"severity": "low", "likelihood": "rare"}


@pytest.mark.parametrize("padding", [0, 1, 3, 10, 30, 100])
def test_a_critical_finding_cannot_be_diluted_by_trivial_ones(padding):
    """The whole point. Old code: 1.000 -> 0.273 -> 0.119 -> 0.062 as padding grew."""
    result = calculate_risk_score([CRITICAL] + [TRIVIAL] * padding)
    assert result["rating"] == "critical"
    assert result["score"] == 1.0


def test_the_score_is_monotone_in_added_risk():
    """Adding a risk must never lower the score. It always could, before."""
    register = []
    previous = -1.0
    for item in [TRIVIAL, MEDIUM, TRIVIAL, HIGH, TRIVIAL, TRIVIAL, CRITICAL, TRIVIAL]:
        register.append(item)
        score = calculate_risk_score(register)["score"]
        assert score >= previous, f"score fell to {score} after adding {item}"
        previous = score


def test_padding_the_register_cannot_improve_the_rating():
    """An analyst must not be able to talk a rating down by listing more items."""
    honest = calculate_risk_score([CRITICAL, HIGH])
    padded = calculate_risk_score([CRITICAL, HIGH] + [TRIVIAL] * 50)
    assert padded["score"] >= honest["score"]
    assert padded["rating"] == honest["rating"] == "critical"


def test_breadth_still_escalates_many_moderate_risks():
    """Peak alone would rate ten high/likely findings the same as one. It must not."""
    one = calculate_risk_score([HIGH])
    ten = calculate_risk_score([HIGH] * 10)
    assert ten["score"] > one["score"]
    assert one["rating"] == "high" and ten["rating"] == "critical"


def test_calibration_table_in_the_docstring_matches_the_code():
    """These are the cases the BREADTH_SCALE constant was chosen to satisfy.
    If you retune the constant, retune this table with it -- do not delete it."""
    expected = [
        ([CRITICAL], 1.000, "critical"),
        ([CRITICAL] + [TRIVIAL] * 30, 1.000, "critical"),
        ([HIGH] * 10, 0.755, "critical"),
        ([HIGH] * 5, 0.562, "high"),
        ([TRIVIAL] * 40, 0.268, "moderate"),
        ([TRIVIAL], 0.031, "low"),
    ]
    for risks, score, rating in expected:
        result = calculate_risk_score(risks)
        assert result["score"] == pytest.approx(score, abs=0.001)
        assert result["rating"] == rating


def test_peak_is_reported_separately_so_the_tail_stays_visible():
    """A reader must be able to see 'the worst single item' without re-deriving it."""
    result = calculate_risk_score([TRIVIAL] * 5 + [CRITICAL])
    assert result["max_item_score"] == 1.0
    assert result["total_risks"] == 6


def test_an_empty_register_is_not_scored_as_safe():
    """No findings means nobody looked, not that there is nothing to find."""
    result = calculate_risk_score([])
    assert result["rating"] == "insufficient_data"
    assert result["score"] == 0.0


def test_unknown_labels_fall_back_to_the_middle_not_to_zero():
    typo = calculate_risk_score([{"severity": "sever", "likelihood": "certain"}])
    assert typo["score"] == pytest.approx(calculate_risk_score([MEDIUM])["score"])
