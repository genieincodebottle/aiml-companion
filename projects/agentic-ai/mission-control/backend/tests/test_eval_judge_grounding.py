"""
Regression tests for the eval plane's grounding check.

The bug
-------
The stub judge decided `grounded` by scanning the narrative prose::

    grounded = not _has(text, "uncited", "unsupported", "grounded=false", ...)

That is a confession detector, not a grounding check. A narrative is ungrounded
because it LACKS citations, not because it announces that it lacks them.
Measured on the old implementation:

    claims=['A is laundering money', 'B is a shell'], evidence=[]  -> score 1.0
    claims=['definitely fraud'],                      evidence=[]  -> score 1.0
    claims=['uncited claim']                                       -> score 0.6

Only the narrative containing the magic word scored below perfect. So in stub
mode -- the default, and what CI runs -- the judge score was 1.00 for anything
the platform could produce, and the promotion gate
(`accuracy >= 1.0 and mean_score >= threshold`) collapsed to its first term.
An eval that cannot return a bad score is not measuring anything.

The judge now reads explicit structural signals computed by the eval, the same
convention the critic agent already used (`every_claim_cited=`). The three
golden alerts still score 1.00 -- they genuinely are grounded, so the fix
discriminates rather than just failing everything.

Run: pytest tests/test_eval_judge_grounding.py -v
"""

from __future__ import annotations

import pytest

from evaluation.run_eval import _citation_gap
from src.control_plane import get_control_plane
from src.models.schemas import JudgeVerdict
from src.tenants.fincrime.stub_brain import synthesize


def _verdict(narrative: dict, matches_expected: bool = True) -> JudgeVerdict:
    """Score a narrative the way evaluation/run_eval.py does."""
    user = (
        f"Narrative: {narrative}\n"
        f"Structural check: uncited_claims={_citation_gap(narrative)} "
        f"evidence_count={len(narrative.get('evidence', []))} "
        f"disposition_matches_expected={str(matches_expected).lower()}"
    )
    return synthesize(JudgeVerdict, "judge system prompt", user)


# === The citation gap itself ===


@pytest.mark.parametrize(
    "claims,evidence,expected_gap",
    [
        (1, 2, 0),  # more evidence than claims
        (2, 2, 0),
        (2, 0, 2),  # nothing cited at all
        (3, 1, 2),
        (0, 0, 0),  # nothing claimed, nothing to cite
    ],
)
def test_citation_gap_counts_uncited_claims(claims, evidence, expected_gap):
    narrative = {"claims": ["c"] * claims, "evidence": [{"id": i} for i in range(evidence)]}
    assert _citation_gap(narrative) == expected_gap


# === The judge must fail ungrounded narratives ===


def test_a_narrative_with_no_evidence_is_not_grounded():
    """Scored 1.0 by the old prose scan."""
    verdict = _verdict({"claims": ["A is laundering money", "B is a shell"], "evidence": []})
    assert verdict.grounded is False
    assert verdict.score < 0.8, "must fall below the default promotion threshold"


def test_more_claims_than_evidence_is_not_grounded():
    verdict = _verdict({"claims": ["a", "b", "c"], "evidence": [{"id": 1}]})
    assert verdict.grounded is False
    assert verdict.score < 0.8


def test_a_properly_cited_narrative_still_scores_full_marks():
    """The check must discriminate. A judge that fails everything is as
    useless as one that passes everything -- it just fails in the safe
    direction, which is how a broken gate survives review."""
    verdict = _verdict({"claims": ["A wired 9k to B"], "evidence": [{"id": 1}, {"id": 2}]})
    assert verdict.grounded is True
    assert verdict.score == 1.0


def test_a_wrong_disposition_is_penalised():
    verdict = _verdict({"claims": ["a"], "evidence": [{"id": 1}]}, matches_expected=False)
    assert verdict.disposition_sound is False
    assert verdict.score < 0.8


def test_the_magic_word_is_no_longer_what_decides_the_score():
    """The old check keyed on the literal string "uncited" appearing in the
    prose. A well-cited narrative that happens to discuss uncited claims must
    not be punished for the vocabulary."""
    verdict = _verdict(
        {
            "claims": ["The prior report contained uncited assertions; this one cites E1."],
            "evidence": [{"id": "E1"}],
        }
    )
    assert verdict.grounded is True


def test_a_caller_supplying_no_structural_signal_cannot_score_a_clean_pass():
    """Ad-hoc callers still get a verdict, but never one that clears the gate
    on evidence nobody checked."""
    verdict = synthesize(JudgeVerdict, "sys", "Narrative: {'claims': ['x'], 'evidence': []}")
    assert verdict.score < 0.8


# === End to end: the golden alerts genuinely pass ===


def test_the_golden_alerts_are_actually_grounded():
    """Guards the other direction: this fix must not have broken the eval."""
    from src.tenants.fincrime.data import SEED_ALERTS

    plane = get_control_plane()
    for row in SEED_ALERTS:
        alert = row["alert"]
        _, result = plane.start_investigation(alert.id)
        narrative = result.state.get("narrative", {})
        if not narrative.get("claims"):
            continue
        assert _citation_gap(narrative) == 0, f"{alert.id}: {_citation_gap(narrative)} uncited claim(s)"
