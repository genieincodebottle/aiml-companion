"""The four multiclass strategies, and the properties that separate them."""
import numpy as np
import pytest

from support_ticket_triage.models.strategies import REGISTRY, build


@pytest.fixture(scope="module")
def trained(fitted, cfg):
    out = {}
    for name in REGISTRY:
        model = build(name, cfg).fit(fitted["Xtr"], fitted["ytr"])
        out[name] = (model, model.predict_proba(fitted["Xte"]))
    return out


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_probabilities_are_a_distribution(name, trained, fitted):
    _, proba = trained[name]
    assert proba.shape == (len(fitted["Xte"]), 6)
    assert (proba >= -1e-9).all(), f"{name} produced a negative probability"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6,
                               err_msg=f"{name} rows do not sum to 1")


def test_submodel_counts_match_the_theory(trained):
    """OvR fits K models, OvO fits K(K-1)/2. That gap is the whole cost story."""
    assert trained["ovr_nb"][0].n_submodels == 6
    assert trained["ovo_nb"][0].n_submodels == 15
    assert trained["native_nb"][0].n_submodels == 1


def test_all_strategies_are_far_better_than_chance(trained, fitted):
    from sklearn.metrics import accuracy_score
    for name, (model, proba) in trained.items():
        pred = model.classes_[np.argmax(proba, axis=1)]
        acc = accuracy_score(fitted["yte"], pred)
        assert acc > 0.60, f"{name} scored {acc:.3f}, barely above guessing"


def test_ovo_votes_are_not_probabilities(trained, fitted, cfg):
    """OvO has no probability model, and the ECE must say so.

    Normalised vote counts look like a distribution and are not one. If this
    test ever starts failing it means someone made OvO look calibrated, which
    would hide the actual lesson.
    """
    from support_ticket_triage.evaluation.metrics import \
        expected_calibration_error
    model, proba = trained["ovo_nb"]
    ece = expected_calibration_error(fitted["yte"].to_numpy(), proba,
                                     model.classes_, cfg.n_calibration_bins)
    assert ece > 0.20, (
        f"OvO ECE is {ece:.3f}, which is suspiciously good for something that "
        "never estimated a probability")


def test_unknown_strategy_is_refused(cfg):
    with pytest.raises(KeyError, match="unknown strategy"):
        build("does_not_exist", cfg)


def test_ovr_handles_an_all_zero_score_row(cfg, fitted):
    """A row every sub-model rejects must not divide by zero."""
    model = build("ovr_nb", cfg).fit(fitted["Xtr"], fitted["ytr"])
    empty = fitted["Xte"].iloc[:5].copy()
    empty.loc[:, :] = 0.0
    proba = model.predict_proba(empty)
    assert np.isfinite(proba).all(), "an empty ticket produced non-finite scores"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
