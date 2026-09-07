"""Tests for the offline judge -- given to you, not part of your engagement.

`eval/judge.py` is the ruler everything else is measured with. If the ruler is
wrong you will chase phantom failures, so these guard it rather than your work.
"""
from eval.judge import HeuristicJudge, MODE_KEYWORDS


def _case(mode="damaged_goods", expected="damaged goods claim"):
    return {"failure_mode": mode, "expected": expected}


def test_mode_matching_is_on_whole_tokens_not_substrings():
    """A tripwire that fires on "opportunity" trains people to ignore it.

    Matching used to be `if keyword in text`, so any word CONTAINING a keyword
    counted. "unit" is in the wrong_address list, which made "opportunity" and
    "community" score as address problems, and classified "shrink wrap failed,
    units spilled across the trailer" as wrong_address with confidence.
    """
    judge = HeuristicJudge()
    for text in ("we saw an opportunity to reduce failed deliveries",
                 "the community centre was closed",
                 "shrink wrap failed, units spilled across the trailer floor"):
        verdict = judge.score(_case(), text)
        assert verdict.predicted_mode != "wrong_address", (
            f"{text!r} was classified as wrong_address on a substring match")


def test_genuine_keyword_evidence_still_matches():
    """Tightening the match must not break the cases it is supposed to catch."""
    judge = HeuristicJudge()
    verdict = judge.score(_case(), "the parcel arrived crushed and wet")
    assert verdict.predicted_mode == "damaged_goods"
    assert verdict.mode_correct


def test_the_judge_admits_what_it_cannot_see():
    """The documented weakness must stay true, or it stops motivating a real judge.

    A plainly-damaged-goods answer carrying none of the keywords scores as no
    mode at all. That is the argument for replacing the heuristic, so the
    example has to keep failing.
    """
    judge = HeuristicJudge()
    verdict = judge.score(_case(), "the pallet was compromised in transit")
    assert verdict.predicted_mode == ""
    assert not verdict.mode_correct


def test_faithfulness_is_clamped_to_a_probability():
    """A judge that returns a bad number you can see beats one that crashes."""
    from eval.judge import Verdict

    assert Verdict(faithfulness=5.0, mode_correct=True).faithfulness == 1.0
    assert Verdict(faithfulness=-2.0, mode_correct=False).faithfulness == 0.0


def test_every_declared_failure_mode_has_keywords():
    """A mode with no keywords can never be predicted, so the gate could not pass."""
    from eval.judge import FAILURE_MODES

    for mode in FAILURE_MODES:
        assert MODE_KEYWORDS.get(mode), f"{mode} has no keyword evidence"
