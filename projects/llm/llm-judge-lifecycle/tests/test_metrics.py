"""Metrics: the definitions, and the ways they are usually got wrong.

Every test here corresponds to a mistake that produces plausible numbers rather
than an error, which is why they are worth pinning. A metric bug does not crash;
it reports a judge as better than it is and nothing else in the system disagrees.
"""

from __future__ import annotations

import pytest

from src.metrics import FAIL, PASS, Judgement, compute, confusion, weighted_score, wilson


def j(human, judge, *, agrees=None, rationale="because"):
    return Judgement(
        example_id=f"{human}-{judge}-{agrees}",
        human_label=human,
        judge_label=judge,
        human_rationale=rationale,
        judge_reason="reason",
        rationale_agrees=agrees,
    )


class TestDefinitions:
    def test_specificity_is_fail_recall(self):
        rows = [j(FAIL, FAIL), j(FAIL, FAIL), j(FAIL, PASS), j(PASS, PASS)]
        assert compute(rows).specificity == pytest.approx(2 / 3)

    def test_recall_is_pass_recall(self):
        rows = [j(PASS, PASS), j(PASS, FAIL), j(FAIL, FAIL)]
        assert compute(rows).recall == pytest.approx(1 / 2)

    def test_reasoning_agreement_divides_by_all_human_fails(self):
        """Eq. 3, and the denominator is the whole point.

        Two judges: one catches a single failure and explains it perfectly, the
        other catches three of four and explains all three. Dividing by
        AGREED-fails scores the first 1.0 and the second 1.0, which says they
        are equally good. Dividing by HUMAN-fails scores them 0.25 and 0.75,
        which is the truth.
        """
        catches_one = [j(FAIL, FAIL, agrees=True)] + [j(FAIL, PASS) for _ in range(3)]
        catches_three = [j(FAIL, FAIL, agrees=True) for _ in range(3)] + [j(FAIL, PASS)]

        assert compute(catches_one).reasoning_agreement == pytest.approx(0.25)
        assert compute(catches_three).reasoning_agreement == pytest.approx(0.75)

    def test_unassessed_rationales_are_not_counted_as_disagreement(self):
        """None means "not assessed" and must not be read as False.

        Conflating them makes the vanilla ablation arm - which never runs the
        meta-judge - score zero on reasoning, and the difference then looks like
        a quality gap when it is a measurement gap.
        """
        rows = [j(FAIL, FAIL, agrees=None), j(FAIL, FAIL, agrees=True)]
        metrics = compute(rows)
        assert metrics.n_assessed_rationales == 1
        assert metrics.reasoning_agreement == pytest.approx(0.5)


class TestEmptyDenominators:
    def test_missing_class_gives_none_not_zero(self):
        """No human failures means specificity is UNDEFINED, not 0.0.

        Returning 0.0 would tell an early-stopping rule the judge is failing
        badly on a split where it was never tested, and the run would keep
        iterating against nothing.
        """
        metrics = compute([j(PASS, PASS), j(PASS, PASS)])
        assert metrics.specificity is None
        assert metrics.recall == pytest.approx(1.0)

    def test_none_never_clears_a_target(self):
        metrics = compute([j(PASS, PASS)])
        assert not metrics.clears({"specificity": 0.5})

    def test_weighted_score_treats_none_as_zero(self):
        """So a rubric producing no measurable signal cannot tie one that does."""
        measurable = compute([j(FAIL, FAIL, agrees=True), j(PASS, PASS)])
        unmeasurable = compute([j(PASS, PASS)])
        assert measurable.weighted > unmeasurable.weighted


class TestWeighting:
    def test_specificity_outweighs_recall_three_to_one(self):
        """The asymmetry priced in Eq. 4: a bad artefact served is not the same
        error as a good one rejected."""
        weights = {"specificity": 3.0, "recall": 1.0, "reasoning_agreement": 1.0}
        catches_bad = compute(
            [j(FAIL, FAIL, agrees=True), j(PASS, FAIL)], weights
        )
        keeps_good = compute([j(FAIL, PASS), j(PASS, PASS)], weights)
        assert catches_bad.weighted > keeps_good.weighted

    def test_weights_are_applied_not_ignored(self):
        metrics = compute([j(FAIL, FAIL, agrees=True), j(PASS, PASS)])
        assert weighted_score(metrics, {"specificity": 3.0}) == pytest.approx(3.0)
        assert weighted_score(metrics, {"specificity": 1.0}) == pytest.approx(1.0)


class TestWilson:
    def test_small_samples_produce_wide_intervals(self):
        """5 of 6 is not evidence of 0.83. The interval has to say so."""
        low, high = wilson(5, 6)
        assert low < 0.5 and high > 0.95

    def test_perfect_score_does_not_assert_certainty(self):
        """The normal approximation returns [1.0, 1.0] here, claiming certainty
        from eight observations. Wilson does not."""
        low, high = wilson(8, 8)
        assert low < 0.75
        assert high == pytest.approx(1.0)

    def test_empty_sample_is_not_an_error(self):
        assert wilson(0, 0) == (0.0, 0.0)

    def test_interval_narrows_as_n_grows(self):
        small = wilson(8, 10)
        large = wilson(800, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestConfusion:
    def test_false_pass_and_false_fail_are_distinguished(self):
        """They are not symmetric and must never be summed into one error count.
        A false pass reaches a user; a false fail costs a regeneration."""
        rows = [j(FAIL, PASS), j(PASS, FAIL), j(PASS, FAIL)]
        counts = confusion(rows)
        assert counts["false_pass"] == 1
        assert counts["false_fail"] == 2
