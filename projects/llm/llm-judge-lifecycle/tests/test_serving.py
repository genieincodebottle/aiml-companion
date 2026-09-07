"""Phase III: the gate, the critic, and the asymmetry that justifies dropping.

The most important test in this file is the first one. If the loop ever serves
an artefact that failed a must-have criterion, the gate is decorative and every
other number in the project describes a system that does not exist.
"""

from __future__ import annotations

import pytest

from src.judge import Judge, JudgePanel, Verdict
from src.serving import (
    DROPPED,
    SERVED_CLEAN,
    SERVED_REVISED,
    ServedResult,
    ServingLoop,
    retry_curve,
)


@pytest.fixture()
def loop(context):
    return ServingLoop(
        context.runtime, context.domain, context.load_rubrics(), context.config.serving
    )


class TestTheGateActuallyGates:
    def test_a_served_artefact_passed_every_must_have_criterion(self, loop, context):
        """The load-bearing assertion of Phase III."""
        for record in context.domain.records.values():
            result = loop.serve(record)
            if result.outcome == DROPPED:
                continue
            verdicts = loop.panel.judge_all(record, result.artefact)
            assert not loop.panel.gate_failures(verdicts), (
                f"{record.id} was served with a gate failure"
            )

    def test_a_soft_failure_does_not_block_serving(self, context):
        """A gate that rejects for length drops accurate, useful work over a
        style preference."""
        panel = JudgePanel(context.runtime, context.domain, context.load_rubrics())
        soft = Verdict("FAIL", "too long", "too_long", "concise")
        assert panel.gate_failures({"concise": soft}) == []
        assert panel.soft_failures({"concise": soft}) == [soft]

    def test_dropping_is_the_default_when_the_budget_runs_out(self, context):
        """One `if`, and the whole design rests on it: a bad artefact served
        cannot be recalled, a missing one costs an opportunity."""
        assert context.config.serving["on_budget_exhausted"] == "drop"

    def test_a_dropped_result_carries_no_artefact(self, loop, context):
        results = [loop.serve(r) for r in context.domain.records.values()]
        for result in results:
            if result.outcome == DROPPED:
                assert result.artefact is None


class TestRevision:
    def test_the_first_draft_is_not_counted_as_a_revision(self, loop, context):
        """Off-by-one here shifts every point on the retry curve one place left
        and makes K=3 look like it buys more than it does."""
        record = next(iter(context.domain.records.values()))
        result = loop.serve(record, max_retries=0)
        assert len(result.attempts) == 1
        assert result.revisions == 0

    def test_a_clean_first_draft_is_labelled_as_such(self, loop, context):
        results = [loop.serve(r) for r in context.domain.records.values()]
        for result in results:
            if result.outcome == SERVED_CLEAN:
                assert result.revisions == 0
            if result.outcome == SERVED_REVISED:
                assert result.revisions >= 1

    def test_the_rejection_reason_is_handed_to_the_generator(self, loop):
        """The critic role. Without the failure mode alongside the prose, the
        writer may rewrite the whole line and lose the parts that passed."""
        failures = [Verdict("FAIL", "generic praise", "generic_filler", "specific")]
        critique = loop._critique(failures)
        assert "generic_filler" in critique
        assert "generic praise" in critique

    def test_more_budget_never_lowers_the_pass_rate(self, loop, context):
        """Monotonicity. A non-monotone curve means state is leaking between
        attempts, and the curve stops being readable as evidence for K."""
        records = list(context.domain.records.values())[:8]
        rates = []
        for k in (0, 2, 4):
            results = [loop.serve(r, max_retries=k) for r in records]
            served = sum(1 for r in results if r.outcome != DROPPED)
            rates.append(served / len(results))
        assert rates == sorted(rates)


class TestRetryCurve:
    def _result(self, outcome, revisions):
        return ServedResult(
            record_id="r",
            outcome=outcome,
            artefact="a" if outcome != DROPPED else None,
            attempts=[None] * (revisions + 1),  # type: ignore[list-item]
        )

    def test_the_curve_is_cumulative(self):
        results = [
            self._result(SERVED_CLEAN, 0),
            self._result(SERVED_REVISED, 1),
            self._result(SERVED_REVISED, 3),
            self._result(DROPPED, 3),
        ]
        curve = retry_curve(results, 3)
        assert [point["passed"] for point in curve] == [1, 2, 2, 3]

    def test_k_zero_is_the_generators_unaided_rate(self):
        """The diagnostic that separates a generator regression from judge
        drift. Both look identical in an aggregate pass rate."""
        results = [self._result(SERVED_CLEAN, 0), self._result(SERVED_REVISED, 2)]
        assert retry_curve(results, 2)[0]["cumulative_pass_rate"] == pytest.approx(0.5)

    def test_dropped_results_never_count_as_passes(self):
        curve = retry_curve([self._result(DROPPED, 3)], 3)
        assert all(point["passed"] == 0 for point in curve)

    def test_an_empty_run_returns_an_empty_curve(self):
        assert retry_curve([], 3) == []


class TestJudgeParsing:
    def _judge(self, context):
        return Judge(
            context.runtime, context.domain, context.domain.criterion("specific")
        )

    def test_unparseable_output_fails_closed(self, context):
        """A guardrail that fails OPEN stops working exactly when it breaks, and
        nothing in the metrics shows it because the artefacts were never judged.
        """
        verdict = self._judge(context)._parse("not json at all")
        assert verdict.label == "FAIL"
        assert verdict.failure_mode == "judge_error"

    def test_a_label_outside_the_vocabulary_fails_closed(self, context):
        """Not every server enforces a schema; several implement only
        `json_object`, which constrains syntax and not shape."""
        verdict = self._judge(context)._parse('{"label": "Maybe", "reason": "hmm"}')
        assert verdict.label == "FAIL"

    def test_a_fail_always_carries_a_reason(self, context):
        """In Phase III the reason IS the revision instruction. An empty one
        burns a retry and changes nothing."""
        verdict = self._judge(context)._parse('{"label": "FAIL", "reason": ""}')
        assert verdict.reason

    def test_a_valid_verdict_round_trips(self, context):
        verdict = self._judge(context)._parse(
            '{"label": "PASS", "reason": "fine", "failure_mode": null}'
        )
        assert verdict.passed and verdict.reason == "fine"


class TestBudget:
    def test_the_cap_is_checked_before_the_call_not_after(self, context):
        """Checking afterwards means the call that crossed the line has already
        been paid for."""
        from src.runtime import BudgetExceeded, Runtime

        runtime = Runtime(context.config, max_usd=0.0)
        runtime.usage.by_role["judge"] = {
            "calls": 1, "input_tokens": 0, "output_tokens": 0, "usd": 1.0,
        }
        with pytest.raises(BudgetExceeded, match="judge"):
            runtime.call("judge", "prompt")
