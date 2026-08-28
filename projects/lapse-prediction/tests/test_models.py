"""Contract tests every model in the zoo must satisfy, plus the properties the
business relies on (valid distributions, monotone CDFs, beating the prior)."""
import numpy as np
import pytest

from lapse_prediction.config import CFG
from lapse_prediction.evaluation.report import monotonicity_violation
from lapse_prediction.features.labels import time_split
from lapse_prediction.models.base import from_cdf
from lapse_prediction.models.ordinal import OrdinalChain
from lapse_prediction.models.zoo import REGISTRY

FAST = ["prior", "logit", "ordinal_chain", "lgbm_multiclass", "xgb_aft", "cox_ph"]


@pytest.fixture(scope="module")
def splits(modelling_table):
    return time_split(modelling_table)


@pytest.mark.parametrize("cls", [c for c in REGISTRY if c.name in FAST],
                         ids=[c.name for c in REGISTRY if c.name in FAST])
def test_model_contract(cls, splits):
    train, test, valid = splits
    m = cls().fit(train, valid=test)
    p = m.predict_proba(valid)

    assert p.shape == (len(valid), CFG.n_classes)
    assert np.isfinite(p).all(), "non-finite probabilities"
    assert (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert monotonicity_violation(p) == 0.0


@pytest.mark.parametrize("cls", [c for c in REGISTRY if c.name in FAST if c.name != "prior"],
                         ids=[c.name for c in REGISTRY if c.name in FAST if c.name != "prior"])
def test_model_beats_the_prior_baseline(cls, splits):
    from sklearn.metrics import roc_auc_score
    train, test, valid = splits
    p = cls().fit(train, valid=test).predict_proba(valid)
    auc = roc_auc_score(valid["lapsed"], p[:, CFG.lapse_index])
    assert auc > 0.6, f"{cls.name} scored {auc:.3f} -- no better than guessing"


def test_from_cdf_enforces_monotonicity():
    F = np.array([[0.5, 0.2, 0.8, 0.9]])      # deliberately non-monotone
    p = from_cdf(F)
    assert (p >= 0).all()
    np.testing.assert_allclose(p.sum(), 1.0)
    assert monotonicity_violation(p) == 0.0


def test_ordinal_chain_is_deterministic(splits):
    train, test, valid = splits
    a = OrdinalChain().fit(train, valid=test).predict_proba(valid)
    b = OrdinalChain().fit(train, valid=test).predict_proba(valid)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_expected_days_lies_inside_the_grace_window(splits):
    from lapse_prediction.evaluation.metrics import expected_days
    train, test, valid = splits
    p = OrdinalChain().fit(train, valid=test).predict_proba(valid)
    eta = expected_days(p)
    assert np.isfinite(eta).all()
    assert (eta >= 0).all() and (eta <= CFG.grace_days).all()


def test_recommended_model_can_actually_be_calibrated(splits):
    """Regression test for a real defect: the training pipeline used to skip
    calibration silently when the model had no calibrate(), which meant the
    RECOMMENDED model shipped uncalibrated while the docs promised otherwise."""
    train, test, valid = splits
    m = OrdinalChain().fit(train, valid=test)
    assert hasattr(m, "calibrate"), "the default model must support calibration"
    assert not m.is_calibrated

    m.calibrate(test)
    assert m.is_calibrated
    p = m.predict_proba(valid)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert monotonicity_violation(p) == 0.0


def test_train_pipeline_refuses_to_silently_skip_calibration():
    """calibrate=True on a model that cannot calibrate must raise, not shrug."""
    import pandas as pd
    from lapse_prediction.pipelines import train as train_mod

    class Uncalibratable:
        def fit(self, tr, valid=None):
            return self

        def predict_proba(self, df):
            return np.zeros((len(df), CFG.n_classes))

    original = train_mod.MODELS.copy()
    train_mod.MODELS["_dummy"] = Uncalibratable
    try:
        with pytest.raises(TypeError, match="cannot be calibrated"):
            train_mod.run("_dummy", n_policies=200, calibrate=True, persist=False)
    finally:
        train_mod.MODELS.clear()
        train_mod.MODELS.update(original)
