"""The generator must actually plant what it claims to plant."""
from pathlib import Path

import pytest

from support_ticket_triage.data.generate import (CLASS_PRIORS, CLASSES,
                                                 DEPENDENT_CLUSTERS,
                                                 DEPENDENT_PAIRS,
                                                 SIGNAL_TOKENS, VOCAB, generate)
from support_ticket_triage.evaluation.independence import planted_pair_report


def test_dependency_strength_zero_makes_the_assumption_true(independent_tickets):
    """At strength 0 the planted pairs must show a lift of about 1.

    This is the control for the whole project. If the lift is not ~1 here then
    the generator has an unintended dependence and every later measurement is
    contaminated.
    """
    report = planted_pair_report(independent_tickets, DEPENDENT_PAIRS)
    assert abs(report["lift"].median() - 1.0) < 0.03, (
        "with dependency_strength=0 the planted pairs should be independent, "
        f"but the median lift is {report['lift'].median():.3f}")
    assert report["lift"].between(0.88, 1.12).all(), (
        "individual lifts strayed further than sampling noise explains: "
        f"{report['lift'].min():.3f} to {report['lift'].max():.3f}")


def test_dependency_strength_high_breaks_the_assumption(tickets):
    """At the configured strength the pairs must be visibly dependent."""
    report = planted_pair_report(tickets, DEPENDENT_PAIRS)
    assert report["lift"].median() > 1.4, (
        "the planted couplings are too weak to teach anything; "
        f"median lift is only {report['lift'].median():.3f}")


def test_every_cluster_token_is_a_real_vocabulary_token():
    """A typo in a cluster would silently plant nothing at all."""
    for cls, anchor, partners in DEPENDENT_CLUSTERS:
        assert cls in CLASSES, f"{cls} is not a class"
        assert anchor in SIGNAL_TOKENS[cls], f"{anchor} is not a {cls} token"
        for p in partners:
            assert p in SIGNAL_TOKENS[cls], f"{p} is not a {cls} token"
            assert p != anchor, "a token cannot be its own redundant partner"


def test_class_priors_are_a_distribution_with_a_rare_class():
    assert abs(sum(CLASS_PRIORS.values()) - 1.0) < 1e-9
    assert min(CLASS_PRIORS.values()) <= 0.05, (
        "the imbalance lesson needs a genuinely rare class")


def test_rare_class_survives_generation(tickets):
    share = tickets["category"].value_counts(normalize=True)
    assert share.min() > 0.01, "the rare class vanished; the lesson goes with it"
    assert set(share.index) == set(CLASSES), "a class is missing entirely"


def test_boilerplate_carries_no_class_signal(independent_tickets):
    """Politeness must be uninformative, or it is not boilerplate.

    Its rate should be roughly equal across classes. A model leaning on `thanks`
    would be learning an artefact of the generator rather than the problem.
    """
    from support_ticket_triage.data.generate import BOILERPLATE
    by_class = independent_tickets.groupby("category")[list(BOILERPLATE)].mean()
    spread = (by_class.max() - by_class.min()).max()
    # 0.06 at 20,000 tickets. The rare class holds ~600 rows, so the standard
    # error on a 0.18 rate is about 0.016 and a max-minus-min across 54 cells
    # runs to roughly three of those.
    assert spread < 0.06, (
        f"boilerplate rate varies by {spread:.3f} across classes, so it leaks "
        "class information and is not boilerplate")


def test_generation_is_reproducible():
    a = generate(n_tickets=500, seed=7)
    b = generate(n_tickets=500, seed=7)
    assert a.equals(b), "same seed must give the same inbox"


def test_tiny_n_is_refused():
    with pytest.raises(ValueError, match="too small"):
        generate(n_tickets=10)


def test_answer_key_never_reaches_fitting_code():
    """The clusters may be read by the dependence report and by tests, and by
    nothing that fits or transforms features.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "support_ticket_triage"
    allowed = {"generate.py", "independence.py", "compare.py"}
    offenders = []
    for f in root.rglob("*.py"):
        if f.name in allowed:
            continue
        text = f.read_text(encoding="utf-8")
        if "DEPENDENT_CLUSTERS" in text or "DEPENDENT_PAIRS" in text:
            offenders.append(f.relative_to(root).as_posix())
    assert not offenders, (
        f"the answer key reached {offenders}; a model must not be able to see "
        "which tokens were wired together")


def test_vocab_has_no_duplicates_and_covers_every_class():
    assert len(VOCAB) == len(set(VOCAB))
    for cls, toks in SIGNAL_TOKENS.items():
        assert set(toks) <= set(VOCAB), f"{cls} has tokens outside the vocabulary"
