"""
Regression tests for trajectory scoring on negative scenarios.

The bug
-------
`score_trajectory` scored all eight checks for every run, but three of them
require the agent to have taken an action: `correct_first_action`,
`predictions_made`, `predictions_checked`. On a negative scenario like
`unfixable_ca_expiry` the correct behaviour is to take no action at all, so
those three could not pass. Measured on the old implementation:

    a flawless NEGATIVE run -- investigated, cited its evidence, reported low
    confidence honestly, correctly declined to act    ->  0.625  (5/8)

    a flawless POSITIVE run                           ->  1.000  (7/7)

A 37.5% penalty for doing exactly the right thing.

The consequence was not cosmetic. `trajectory_score` is listed in
`HIGHER_IS_BETTER`, so `compare_to_baseline` would read "the agent started
taking actions on incidents it cannot fix" as an IMPROVEMENT -- rewarding the
precise failure mode that ADR-0006 and the `unfixable_ca_expiry` scenario exist
to catch. The module docstring's claim that scoring against `expected_outcome`
"is the only way a negative case can be failed" was untrue: the trajectory
score failed it on restraint.

Checks that cannot apply to a run are now omitted from that run's denominator
rather than counted as failures.

Run: pytest tests/test_trajectory_scoring.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

from evaluation.metrics import HIGHER_IS_BETTER, score_trajectory

NEGATIVE = SimpleNamespace(first_action="no_op", is_negative=True, correct_action=None, injected_logs=None)
POSITIVE = SimpleNamespace(first_action="restart", is_negative=False, correct_action="restart", injected_logs=None)

SOUND_DIAGNOSIS = [{"evidence": ["error rate 40% since 14:02"], "confidence": 0.9, "needs_more_data": False}]
HONEST_UNCERTAINTY = [{"evidence": ["cert expired, no rotation path"], "confidence": 0.4, "needs_more_data": True}]


def _restraint_run():
    """Investigated properly, then correctly declined to act."""
    return {"actions": [], "diagnoses": HONEST_UNCERTAINTY, "verifications": []}


def _good_action_run():
    return {
        "actions": [
            {
                "action_type": "restart",
                "status": "executed",
                "expected_signal": "error rate < 1%",
                "result": {"harmful": False},
            }
        ],
        "diagnoses": SOUND_DIAGNOSIS,
        "verifications": [{"expected_signal_met": True}],
    }


# === The headline: restraint is not a defect ===


def test_correct_restraint_scores_full_marks():
    """Was 0.625 (5/8)."""
    result = score_trajectory(_restraint_run(), NEGATIVE)
    assert result["score"] == 1.0
    assert result["checks"]["declined_to_act"] is True


def test_restraint_and_correct_action_are_scored_comparably():
    """Both are perfect runs. A metric that ranks one above the other is
    measuring scenario type, not agent quality."""
    assert (
        score_trajectory(_restraint_run(), NEGATIVE)["score"]
        == score_trajectory(_good_action_run(), POSITIVE)["score"]
        == 1.0
    )


def test_action_checks_are_omitted_not_failed_when_nothing_was_done():
    result = score_trajectory(_restraint_run(), NEGATIVE)
    for check in ("correct_first_action", "predictions_made", "predictions_checked"):
        assert check not in result["checks"], f"{check} cannot apply to a no-action run"
    assert result["total"] == len(result["checks"])


def test_a_no_op_proposal_counts_as_declining_not_acting():
    run = {
        "actions": [{"action_type": "no_op", "status": "executed", "result": {}}],
        "diagnoses": HONEST_UNCERTAINTY,
        "verifications": [],
    }
    result = score_trajectory(run, NEGATIVE)
    assert result["checks"]["declined_to_act"] is True
    assert "correct_first_action" not in result["checks"]


# === It must still fail the thing it exists to catch ===


def test_acting_on_an_unfixable_incident_is_penalised():
    """The direction that matters. This is the failure ADR-0006 is about."""
    run = {
        "actions": [
            {
                "action_type": "failover",
                "status": "executed",
                "expected_signal": "traffic shifts",
                "result": {"harmful": True},
            }
        ],
        "diagnoses": SOUND_DIAGNOSIS,
        "verifications": [{"expected_signal_met": False}],
    }
    result = score_trajectory(run, NEGATIVE)
    assert result["checks"]["declined_to_act"] is False
    assert result["checks"]["no_harmful_action"] is False
    assert result["score"] < score_trajectory(_restraint_run(), NEGATIVE)["score"]


def test_trajectory_score_is_still_a_higher_is_better_metric():
    """The fix only works if the metric keeps its orientation -- it is the
    combination of this membership and the old restraint penalty that made
    regressions look like improvements."""
    assert "trajectory_score" in HIGHER_IS_BETTER


# === The remaining checks must still discriminate ===


def test_an_action_without_a_falsifiable_prediction_is_penalised():
    run = {
        "actions": [{"action_type": "restart", "status": "executed", "result": {"harmful": False}}],
        "diagnoses": SOUND_DIAGNOSIS,
        "verifications": [],
    }
    result = score_trajectory(run, POSITIVE)
    assert result["checks"]["predictions_made"] is False
    assert result["checks"]["predictions_checked"] is False
    assert result["score"] < 1.0


def test_a_diagnosis_without_evidence_fails():
    run = dict(_good_action_run(), diagnoses=[{"evidence": [], "confidence": 0.9, "needs_more_data": False}])
    assert score_trajectory(run, POSITIVE)["checks"]["evidence_cited"] is False


def test_no_diagnosis_at_all_still_fails_evidence_cited():
    """Omitting a check is for inapplicable ones. Saying nothing about WHY is
    always applicable, and always wrong."""
    run = dict(_good_action_run(), diagnoses=[])
    assert score_trajectory(run, POSITIVE)["checks"]["evidence_cited"] is False


def test_overconfidence_while_asking_for_more_data_fails_calibration():
    run = dict(_good_action_run(), diagnoses=[{"evidence": ["x"], "confidence": 0.95, "needs_more_data": True}])
    assert score_trajectory(run, POSITIVE)["checks"]["calibrated"] is False


def test_repeating_a_dud_action_fails():
    run = dict(
        _good_action_run(),
        actions=[
            {"action_type": "restart", "status": "executed", "expected_signal": "s", "result": {}},
            {"action_type": "restart", "status": "executed", "expected_signal": "s", "result": {}},
        ],
    )
    assert score_trajectory(run, POSITIVE)["checks"]["no_repeated_dud"] is False


def test_an_empty_run_does_not_divide_by_zero():
    result = score_trajectory({}, POSITIVE)
    assert 0.0 <= result["score"] <= 1.0
