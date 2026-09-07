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


def test_train_pipeline_refuses_to_silently_skip_calibration(tmp_path):
    """calibrate=True on a model that cannot calibrate must raise, not shrug.

    Runs against a tmp_path config: the default CFG points at the project's real
    data/ directory, and a test that writes a 200-policy ledger there would
    poison the cache every documented command reads afterwards.
    """
    import dataclasses

    import pandas as pd
    from lapse_prediction.pipelines import train as train_mod

    cfg = dataclasses.replace(
        CFG, raw_data=str(tmp_path / "dues.parquet"),
        modelling_table=str(tmp_path / "modelling.parquet"),
        model_store=str(tmp_path / "models"), artifacts=str(tmp_path / "artifacts"))

    class Uncalibratable:
        def fit(self, tr, valid=None):
            return self

        def predict_proba(self, df):
            return np.zeros((len(df), CFG.n_classes))

    original = train_mod.MODELS.copy()
    train_mod.MODELS["_dummy"] = Uncalibratable
    try:
        with pytest.raises(TypeError, match="cannot be calibrated"):
            train_mod.run("_dummy", cfg, n_policies=200, calibrate=True,
                          persist=False)
    finally:
        train_mod.MODELS.clear()
        train_mod.MODELS.update(original)


def test_calibration_does_not_wreck_the_ranking(splits):
    """The retention queue is a RANKING, and calibration must not reorder it.

    Regression test for a defect that shipped: the default calibrator was
    isotonic, which is a step function. On this cohort it collapsed ~3,400
    distinct scores into 32 levels, and the ties cost 0.026 PR-AUC -- roughly
    8% of the model's ranking power -- while making the Brier score and ECE
    worse too. Every column moved the wrong way.

    Platt is strictly monotone, so it cannot reorder anything. That is the
    property being pinned here: whatever calibrator is configured, it must not
    cost meaningful ranking on a cohort it never saw.
    """
    from sklearn.metrics import average_precision_score
    from lapse_prediction.features.labels import split_oot_cohort

    train, test, valid = splits
    early_stop, calib = split_oot_cohort(test, CFG)
    assert len(set(early_stop.index) & set(calib.index)) == 0, (
        "early stopping and calibration must not share rows")

    m = OrdinalChain().fit(train, valid=early_stop)
    y = valid["lapsed"].to_numpy()
    before = average_precision_score(y, m._raw_proba(valid)[:, CFG.lapse_index])
    m.calibrate(calib)
    after = average_precision_score(y, m.predict_proba(valid)[:, CFG.lapse_index])

    assert after >= before - 0.005, (
        f"calibration cost {before - after:.4f} PR-AUC; a calibrator that "
        f"reorders the retention queue is not repairing the model")


def test_calibration_audit_reports_both_axes(splits):
    """The audit is what stops calibration being shipped on faith."""
    from lapse_prediction.models.base import calibration_audit
    from lapse_prediction.features.labels import split_oot_cohort

    train, test, valid = splits
    early_stop, calib = split_oot_cohort(test, CFG)
    m = OrdinalChain().fit(train, valid=early_stop).calibrate(calib)
    audit = calibration_audit(m, valid, CFG)

    assert list(audit["scores"]) == ["raw", CFG.calibration_method]
    for col in ("lapse_pr_auc", "brier", "ece", "distinct_scores"):
        assert col in audit.columns

    # Deliberately NOT asserted: that calibration improved the ECE. On this
    # 400-policy fixture the calibration cohort is ~100 rows and it often does
    # not -- which is the finding, not a flaky test. Baking "calibration helps"
    # into an assertion would re-commit the exact error this audit exists to
    # catch. What IS guaranteed, at any sample size, is that a strictly
    # monotone map cannot reorder the queue:
    assert audit["distinct_scores"].iloc[1] >= audit["distinct_scores"].iloc[0] * 0.99
