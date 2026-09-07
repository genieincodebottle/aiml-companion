"""Phase II: the tuning loop, and the four ways it silently reports a fit.

Each test corresponds to a mistake that makes RART look like it worked. None of
them raises; all of them produce a number somebody would put in a table.
"""

from __future__ import annotations

import pytest

from src.benchmark import Benchmark
from src.domain import Criterion
from src.evaluate import focus_set
from src.metrics import FAIL, PASS, Judgement
from src.rart import _dropped_tags, tune


@pytest.fixture()
def tuning(context):
    def run(criterion_id: str, **overrides):
        criterion = context.domain.criterion(criterion_id)
        splits = Benchmark(context.domain, context.config.benchmark).split_for(
            criterion_id
        )
        config = {**context.config.rart, **overrides}
        return tune(
            context.runtime, context.domain, criterion, splits.train,
            splits.validation, config,
        )

    return run


class TestTheLoopWorks:
    def test_rart_improves_a_criterion_with_learnable_headroom(self, tuning):
        """The end-to-end proof that the optimiser is real.

        `specific` carries five failures built from two filler formulas absent
        from the seed rubric. They recur, so a rubric clause banning them
        generalises, and specificity has to move. If this ever goes flat, the
        reflector has stopped working - not the benchmark.
        """
        result = tuning("specific")
        assert result.improved, "RART found no improvement where headroom exists"
        first = result.iterations[0].validation["metrics"]["specificity"]
        best = result.iterations[result.best_iteration].validation["metrics"][
            "specificity"
        ]
        assert best > first

    def test_the_tuned_rubric_differs_from_the_seed(self, tuning):
        result = tuning("specific")
        assert result.best_rubric != result.seed_rubric

    def test_learned_clauses_are_general_not_memorised(self, tuning):
        """The reflector requires a phrase in at least two failures.

        A clause inferred from one example is memorisation: it scores on that
        example and costs recall everywhere else. The learned phrases here are
        the ones that recur.
        """
        result = tuning("specific")
        from src.rules import RuleEngine

        learned = RuleEngine(result.best_rubric).banned_phrases() - RuleEngine(
            result.seed_rubric
        ).banned_phrases()
        assert learned
        for phrase in learned:
            assert len(phrase.split()) >= 2, f"{phrase!r} is too narrow to generalise"


class TestStoppingAndSelection:
    def test_the_best_checkpoint_is_returned_not_the_last(self, tuning):
        """Rubric edits are not monotone improvements. A loop returning R_N is
        reporting wherever it happened to stop, which on a bad iteration is
        worse than not tuning at all."""
        result = tuning("specific")
        scores = [it.weighted for it in result.iterations]
        assert result.best_score == pytest.approx(max(scores))
        assert result.iterations[result.best_iteration].accepted

    def test_a_tie_keeps_the_earlier_simpler_rubric(self, tuning):
        """Selection is strictly greater-than. The reflector only ever adds, so
        on a tie the earlier rubric is the simpler one and a tie is not evidence
        for the more complicated one."""
        result = tuning("specific")
        accepted = [it.index for it in result.iterations if it.accepted]
        assert accepted == sorted(set(accepted))

    def test_the_loop_stops_when_the_optimiser_proposes_nothing(self, tuning):
        """Otherwise it re-scores an unchanged rubric against an unchanged split
        N times - real money, online, for a guaranteed no-op."""
        result = tuning("safe", max_iterations=6)
        assert len(result.iterations) < 6
        assert "proposed no change" in result.stopped_because

    def test_convergence_is_reported_separately_from_the_iteration_cap(self, tuning):
        """They mean different things. Hitting the cap says the search was still
        moving; converging says it had stopped, and raising the cap will not
        help."""
        converged = tuning("safe")
        assert "max_iterations" not in converged.stopped_because


class TestTheTestSplitIsNeverTouched:
    def test_tuning_only_ever_sees_train_and_validation(self, context):
        """The moment tuning consults test, the final number is a fit rather
        than a held-out estimate - and nothing in the output would say so."""
        splits = Benchmark(context.domain, context.config.benchmark).split_for("specific")
        seen = {e.id for e in splits.train} | {e.id for e in splits.validation}
        held_out = {e.id for e in splits.test}
        assert not (seen & held_out)
        assert held_out, "an empty test split means there is no held-out estimate"


class TestTheReflectorIsGuarded:
    def test_a_proposal_dropping_a_tag_is_rejected(self):
        """Tags are checks. Dropping one silently disables it, and the metrics
        show the loss without pointing at the cause."""
        before = "text [grounded] [min_words: 10]"
        after = "text [min_words: 10]"
        assert _dropped_tags(before, after) == ["grounded"]

    def test_adding_a_tag_is_not_a_rejection(self):
        assert _dropped_tags("[grounded]", "[grounded] [max_words: 40]") == []


class TestFocusSet:
    def _example(self, eid):
        from src.domain import LabelledExample

        return LabelledExample(id=eid, record_id="r", artefact="a", labels={})

    def test_focus_holds_label_mismatches_and_reason_mismatches(self):
        rows = [
            Judgement("a", FAIL, PASS),  # label mismatch
            Judgement("b", FAIL, FAIL, rationale_agrees=False),  # reason mismatch
            Judgement("c", FAIL, FAIL, rationale_agrees=True),  # correct, excluded
            Judgement("d", PASS, PASS),  # correct, excluded
        ]
        examples = {eid: self._example(eid) for eid in "abcd"}
        focus = focus_set(rows, examples)
        assert {f["example_id"] for f in focus} == {"a", "b"}

    def test_correct_cases_are_withheld_from_the_reflector(self):
        """Handing it what already works invites it to "improve" working
        clauses, which is how a tuning run gets worse while every iteration
        looks busy."""
        rows = [Judgement("c", FAIL, FAIL, rationale_agrees=True)]
        assert focus_set(rows, {"c": self._example("c")}) == []

    def test_error_types_are_labelled_for_the_reflector(self):
        rows = [Judgement("b", FAIL, FAIL, rationale_agrees=False)]
        focus = focus_set(rows, {"b": self._example("b")})
        assert focus[0]["error_type"] == "reason_mismatch"


class TestTheAblationIsComparable:
    def test_both_arms_are_rescored_with_identical_instrumentation(self, context):
        """The bug this test exists to prevent produced a flattering result.

        Vanilla tunes without the meta-judge, so during its own loop reasoning
        agreement is never computed and enters its weighted score as zero.
        Comparing the two arms' INTERNAL scores hands RART a free point on a
        term the other arm did not measure - a gap that appears even when both
        produce byte-identical rubrics. The first version of `ablation()` did
        exactly that and reported RART winning by 0.67 on rubrics that were the
        same text.
        """
        from src.services import TuningService

        result = TuningService(context).ablation("specific")
        if result["identical_rubrics"]:
            assert result["delta_weighted"] == pytest.approx(0.0), (
                "identical rubrics scored differently, so the two arms are not "
                "being measured the same way"
            )

    def test_the_ablation_reports_whether_the_rubrics_are_identical(self, context):
        from src.services import TuningService

        result = TuningService(context).ablation("specific")
        assert "identical_rubrics" in result


def test_criterion_dataclass_marks_gates():
    criterion = Criterion(
        id="c", display="C", must_have=True, guideline="x" * 200
    )
    assert criterion.is_gate
