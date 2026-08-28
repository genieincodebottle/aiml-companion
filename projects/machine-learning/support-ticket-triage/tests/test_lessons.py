"""Every claim the write-up makes, asserted against the code.

If a lesson stops being true after a change, this file fails rather than the
documentation quietly becoming fiction.
"""
import numpy as np

from support_ticket_triage.data.generate import DEPENDENT_PAIRS, generate
from support_ticket_triage.evaluation import independence, metrics
from support_ticket_triage.features.build import split, xy
from support_ticket_triage.models.calibrate import (CalibratedNB,
                                                    ranking_preserved)
from support_ticket_triage.models.strategies import build
from support_ticket_triage.routing import decide


def test_blind_survey_rediscovers_the_planted_pairs(tickets, cfg):
    """LESSON: the dependence measurement works without an answer key.

    This is what earns the right to run the same survey on real data, where
    nobody hands you the list of couplings.
    """
    tokens = [c for c in tickets.columns if c not in
              ("ticket_id", "category", "text", "n_tokens")]
    survey = independence.survey_all_pairs(tickets, tokens, top=40)
    recovery = independence.recovered_planted_pairs(survey, DEPENDENT_PAIRS)
    assert recovery["recall"] >= 0.5, (
        f"the blind survey found only {recovery['found']}/{recovery['planted']} "
        "planted pairs, so it cannot be trusted on data we did not build")


def test_calibration_repairs_probabilities_without_rewriting_decisions(fitted, cfg):
    """LESSON: ranking and probability are separable problems.

    A calibrator is a monotone map, so it may move every probability while
    leaving most decisions alone. That is the whole resolution of the project.
    """
    raw = build("native_nb", cfg).fit(fitted["Xtr"], fitted["ytr"])
    cal = CalibratedNB(cfg).fit(fitted["Xtr"], fitted["ytr"])
    p_raw = raw.predict_proba(fitted["Xte"])
    p_cal = cal.predict_proba(fitted["Xte"])
    y = fitted["yte"].to_numpy()

    ece_raw = metrics.expected_calibration_error(y, p_raw, raw.classes_)
    ece_cal = metrics.expected_calibration_error(y, p_cal, cal.classes_)
    preserved = ranking_preserved(p_raw, p_cal, raw.classes_)

    assert ece_cal < ece_raw, (
        f"calibration made ECE worse ({ece_raw:.4f} -> {ece_cal:.4f})")
    assert preserved["top1_agreement"] > 0.85, (
        "calibration changed too many decisions to be called a monotone repair; "
        f"only {preserved['top1_agreement']:.1%} of top-1 picks survived")


def test_the_assumption_is_false_but_cheap(cfg):
    """LESSON: the headline. False assumption, small cost against a control.

    Naive Bayes assumes independence. Logistic regression does not. Fit both on
    the same dependent data: the gap is what the assumption costs. The bound is
    deliberately generous, because the claim is "cheap", not "free".
    """
    df = generate(n_tickets=4000, seed=cfg.seed, dependency_strength=0.95)
    train, test = split(df, cfg)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    y = yte.to_numpy()

    nb = build("native_nb", cfg).fit(Xtr, ytr)
    lr = build("softmax_lr", cfg).fit(Xtr, ytr)
    acc_nb = metrics.score(y, nb.predict_proba(Xte), nb.classes_)["accuracy"]
    acc_lr = metrics.score(y, lr.predict_proba(Xte), lr.classes_)["accuracy"]

    lift = independence.planted_pair_report(df, DEPENDENT_PAIRS)["lift"].median()
    assert lift > 1.4, "the data is not actually violating the assumption"
    assert acc_lr - acc_nb < 0.06, (
        f"the assumption cost {acc_lr - acc_nb:.4f} accuracy, which is more "
        "than the write-up claims")


def test_raising_the_threshold_trades_volume_for_safety(fitted, cfg):
    """LESSON: the operating curve is monotone, which is why it is usable."""
    cal = CalibratedNB(cfg).fit(fitted["Xtr"], fitted["ytr"])
    proba = cal.predict_proba(fitted["Xte"])
    sweep = decide.threshold_sweep(proba, cal.classes_, fitted["yte"].to_numpy())

    shares = sweep["auto_routed_share"].to_numpy()
    assert (np.diff(shares) <= 1e-9).all(), (
        "auto-routed share must fall as the threshold rises")
    errors = sweep["errors_reaching_customer"].to_numpy()
    assert errors[0] >= errors[-1], (
        "a stricter threshold must not let MORE errors through")


def test_accuracy_hides_the_rare_class(fitted, cfg):
    """LESSON: why macro-F1 and per-class recall exist.

    The rare class must score materially worse than the headline accuracy,
    which is the entire argument for not reporting accuracy alone.
    """
    model = build("native_nb", cfg).fit(fitted["Xtr"], fitted["ytr"])
    proba = model.predict_proba(fitted["Xte"])
    y = fitted["yte"].to_numpy()
    table = metrics.per_class_table(y, proba, model.classes_)
    overall = metrics.score(y, proba, model.classes_)["accuracy"]

    rarest = table.iloc[-1]
    assert rarest["share"] < 0.06, "expected the last row to be the rare class"
    assert rarest["recall"] < overall, (
        "the rare class is not being hurt by imbalance, so the lesson is gone")
