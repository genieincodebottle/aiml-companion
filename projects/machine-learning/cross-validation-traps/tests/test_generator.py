"""The generator is the answer key. If it is wrong, every table is wrong."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cv_traps.data import schema
from cv_traps.data.generate import CUSTOMER_FEATURES, generate
from cv_traps.evaluation.truth import split_panel
from cv_traps.features.build import booster, xy


def test_panel_matches_its_contract(panel):
    schema.validate(panel)


def test_customer_features_are_constant_within_a_customer(panel):
    """The group trap depends on these fingerprinting the customer."""
    for col in CUSTOMER_FEATURES:
        assert panel.groupby("customer_id")[col].nunique().max() == 1


def test_one_row_per_customer_period(panel):
    assert not panel.duplicated(["customer_id", "period"]).any()


def test_the_signal_is_actually_learnable(split, cfg):
    """Guards the failure that wasted the first build of this project.

    An earlier generator had a latent effect larger than every observable
    coefficient combined, so the truth holdout sat at AUC 0.52. Nothing was
    measurable, and worse, nothing LOOKED broken: the tables printed fine and
    every difference in them was noise. A truth this close to chance makes the
    whole project meaningless, so it is asserted rather than assumed.
    """
    dev, out = split["dev"], split["out"]
    X_dev, y_dev = xy(dev, core_only=True)
    X_out, y_out = xy(out, core_only=True)
    model = booster(seed=cfg.seed).fit(X_dev, y_dev)
    auc = roc_auc_score(y_out, model.predict_proba(X_out)[:, 1])
    assert auc > 0.65, (
        f"truth AUC is {auc:.3f}; with a truth this close to chance every "
        "difference between schemes is noise")


def test_turning_the_group_effect_off_removes_the_group_leak(
        exchangeable_panel, cfg):
    """The control. Grouped and ungrouped folds should agree when they may.

    This is the test that makes the project's claim falsifiable: if grouped CV
    were simply always lower, that would be an artefact of grouped folds
    training on fewer distinct customers rather than evidence of a leak.
    """
    from cv_traps.evaluation.metrics import cv_score
    from cv_traps.splitters import schemes

    dev, _ = split_panel(exchangeable_panel, cfg)
    X, y = xy(dev, core_only=True)
    model = booster(seed=cfg.seed)
    a, _ = cv_score(model, X, y,
                    list(schemes.stratified(dev, y, cfg.n_folds, cfg.seed)))
    b, _ = cv_score(model, X, y,
                    list(schemes.grouped(dev, y, cfg.n_folds, cfg.seed)))
    assert abs(a - b) < 0.02, (
        f"with no latent customer effect the two schemes should agree, but "
        f"stratified={a:.4f} and grouped={b:.4f}")


def test_noise_columns_carry_no_signal(panel):
    """If they did, the preprocessing trap would be measuring the wrong thing."""
    noise = [c for c in panel.columns if c.startswith("noise_")]
    y = panel["churned"].to_numpy()
    aucs = np.array([roc_auc_score(y, panel[c]) for c in noise])
    assert abs(aucs.mean() - 0.5) < 0.01
    assert aucs.max() < 0.58, "a noise column carries real signal"


def test_generator_rejects_a_panel_too_small_to_group():
    with pytest.raises(ValueError, match="too small"):
        generate(n_customers=50)
    with pytest.raises(ValueError, match="at least 4"):
        generate(n_customers=300, n_periods=3)
