"""
Regression tests for evaluation sampling.

The bug
-------
`run_evaluator` samples claims for LLM-as-judge review. When a claim was
sampled OUT, it returned::

    {"evaluation_passed": True, "pipeline_trace": [{"status": "skipped_sampling"}]}

"nobody looked at this claim" was recorded as "this claim was judged and
passed". configs/base.yaml sets `batch_eval_sample_rate: 0.10`, so roughly
**90% of auto-processed claims carried a pass they never earned**, and the
batch harness then computed:

    "all_evals_passed": all(r.get("eval_passed", False) for r in successful)

over all of them -- a quality headline made almost entirely of unexamined
claims, and one that could only ever read "everything is fine". The average
score next to it was computed over the evaluated subset only, so the two
numbers did not even share a denominator.

`evaluation_passed` is now three-valued: True (judged, passed), False (judged,
failed), None (never judged). Routing is unchanged -- an unevaluated claim
still continues, which is the point of sampling -- but the aggregate can now
tell the difference.

Run: pytest tests/test_evaluation_sampling.py -v
"""
import pytest

from src.agents.graph import route_after_evaluation


# === The three-valued flag ===

def test_a_judged_and_failed_claim_goes_to_a_human():
    assert route_after_evaluation({"evaluation_passed": False}) == "hitl_checkpoint"


def test_a_judged_and_passed_claim_is_released():
    assert route_after_evaluation({"evaluation_passed": True}) == "communication_agent"


def test_an_unevaluated_claim_continues_rather_than_flooding_hitl():
    """Sampling exists to avoid judging every claim. Routing None to HITL would
    send 90% of claims to a human and defeat it -- which is what the obvious
    `if not evaluation_passed` rewrite would do."""
    assert route_after_evaluation({"evaluation_passed": None}) == "communication_agent"
    assert route_after_evaluation({}) == "communication_agent"


# === The skip path must not fabricate a pass ===

def test_sampled_out_claims_are_not_recorded_as_passing(monkeypatch):
    import src.evaluation.evaluator as ev

    monkeypatch.setattr(ev, "get_evaluation_config",
                        lambda: {"batch_eval_sample_rate": 0.0})   # skip everything
    monkeypatch.setattr("random.random", lambda: 1.0)

    state = {"claim": {"claim_id": "C-1", "estimated_amount": 100}}
    result = ev.run_evaluator(state)

    assert result["evaluation_passed"] is None, \
        "a skipped evaluation must not look like a pass"
    assert result["evaluation_skipped"] is True
    assert result["pipeline_trace"][0]["status"] == "skipped_sampling"


@pytest.mark.parametrize("state_extra", [
    {"hitl_required": True},
    {"human_override": True},
])
def test_claims_that_must_always_be_evaluated_are_not_sampled_out(monkeypatch, state_extra):
    """These bypass the sampler, so they must not take the skip path at all."""
    import src.evaluation.evaluator as ev

    monkeypatch.setattr(ev, "get_evaluation_config",
                        lambda: {"batch_eval_sample_rate": 0.0})
    monkeypatch.setattr("random.random", lambda: 1.0)
    monkeypatch.setattr(ev, "get_judge_llm", lambda: (_ for _ in ()).throw(
        AssertionError("reached the judge -- correct: this claim was not skipped")))

    state = {"claim": {"claim_id": "C-2", "estimated_amount": 100}, **state_extra}
    with pytest.raises(AssertionError, match="not skipped"):
        ev.run_evaluator(state)


# === The batch summary must not count skips as passes ===

def _summarise(successful):
    """Mirrors the aggregation in evaluation/run_eval.py."""
    evaluated = [r for r in successful if r.get("eval_passed") is not None]
    return {
        "evaluated": len(evaluated),
        "skipped_by_sampling": len(successful) - len(evaluated),
        "eval_pass_rate": (
            sum(1 for r in evaluated if r["eval_passed"]) / len(evaluated)
            if evaluated else None),
        "all_evaluated_claims_passed": (
            all(r["eval_passed"] for r in evaluated) if evaluated else None),
    }


def test_the_summary_reports_over_evaluated_claims_only():
    successful = ([{"eval_passed": None}] * 9) + [{"eval_passed": False}]
    summary = _summarise(successful)

    # Old behaviour: the nine skips carried True, so all_evals_passed was
    # decided by 1 real judgement out of 10 rows -- and read as 90% signal.
    assert summary["evaluated"] == 1
    assert summary["skipped_by_sampling"] == 9
    assert summary["eval_pass_rate"] == 0.0
    assert summary["all_evaluated_claims_passed"] is False


def test_a_batch_where_nothing_was_evaluated_reports_none_not_success():
    summary = _summarise([{"eval_passed": None}] * 10)
    assert summary["evaluated"] == 0
    assert summary["eval_pass_rate"] is None
    assert summary["all_evaluated_claims_passed"] is None, \
        "no evidence is not the same as a clean bill of health"
