"""Regression tests on the project's actual claims. If a lesson stops being
true, the README becomes wrong -- so the lessons are asserted, not narrated."""
import numpy as np
import pytest

from rent_price_explainer.evaluation import metrics, recovery
from rent_price_explainer.features.build import TARGET, design_matrix
from rent_price_explainer.models.gbm import GBM
from rent_price_explainer.models.linear import InteractionOLS, NaiveOLS, SpecifiedOLS


@pytest.fixture(scope="module")
def fitted(splits):
    train, test = splits
    return {"naive": NaiveOLS().fit(train), "spec": SpecifiedOLS().fit(train),
            "inter": InteractionOLS().fit(train),
            "gbm": GBM(seed=42).fit(train)}, train, test


def _mape(m, test):
    return metrics.median_ape(test[TARGET], m.predict(test))


def test_specification_beats_naive_by_a_lot(fitted):
    ms, _, test = fitted
    assert _mape(ms["spec"], test) < _mape(ms["naive"], test) * 0.7


def test_r2_never_falls_when_junk_is_added(splits):
    """The claim that makes R² unusable for feature selection."""
    train, _ = splits
    X = design_matrix(train, include_junk=True)
    junk = [c for c in X.columns if c.startswith("junk_")]
    base = [c for c in X.columns if not c.startswith("junk_")]
    demo = metrics.r2_inflation_demo(X, np.log(train[TARGET]), junk, base)
    changes = demo["r2_change"].dropna()
    assert (changes >= -1e-9).all(), "R² fell when junk was added -- impossible"
    assert demo["r2"].iloc[-1] > demo["r2"].iloc[0], "junk did not inflate R²"


def test_adjusted_r2_is_less_fooled_than_r2(splits):
    train, _ = splits
    X = design_matrix(train, include_junk=True)
    junk = [c for c in X.columns if c.startswith("junk_")]
    base = [c for c in X.columns if not c.startswith("junk_")]
    demo = metrics.r2_inflation_demo(X, np.log(train[TARGET]), junk, base)
    assert demo["adj_r2_change"].dropna().sum() < demo["r2_change"].dropna().sum()


def test_specified_ols_recovers_the_truth(fitted):
    ms, _, _ = fitted
    r = recovery.score_recovery(ms["inter"], "interaction_ols")
    s = recovery.recovery_summary(r)
    assert s["mean_abs_pct_error"] < 12, f"recovery drifted: {s}"
    assert s["ci_coverage"] >= 0.6


def test_the_interaction_is_recovered_accurately(fitted):
    """The term the GBM found for us should come back nearly exactly."""
    ms, _, _ = fitted
    r = recovery.score_recovery(ms["inter"], "x")
    row = r[r["term"] == "metro_km_x_premium"].iloc[0]
    assert abs(row["pct_error"]) < 20, f"interaction recovery poor: {row.to_dict()}"


def test_gbm_cannot_report_coefficients(fitted):
    ms, _, _ = fitted
    assert ms["gbm"].coefficients().empty
    assert recovery.score_recovery(ms["gbm"]).iloc[0]["true_beta"] != \
        recovery.score_recovery(ms["gbm"]).iloc[0]["true_beta"]   # NaN


def test_collinearity_widens_the_coefficient_spread(splits):
    """Keeping both area columns must make the estimate less stable."""
    train, _ = splits
    d = recovery.collinearity_damage(train, n_seeds=6).set_index("spec")
    assert d.loc["both areas kept", "std"] > d.loc["twin dropped", "std"] * 2


def test_level_target_model_is_reported_as_not_comparable(fitted):
    ms, _, _ = fitted
    r = recovery.score_recovery(ms["naive"])
    assert "not comparable" in str(r.iloc[0]["note"])


def test_shap_on_a_linear_model_is_just_the_coefficient(fitted):
    """The identity that earns the right to trust SHAP on the tree:
    SHAP_i(x) == coef_i * (x_i - mean(x_i)) for a linear model."""
    from rent_price_explainer.explain import shap_report
    ms, _, test = fitted
    m = ms["inter"]
    table = shap_report.linear_shap_identity(m, test)
    X = m._prepare(test).reindex(columns=m.cols, fill_value=0.0)
    for row in table.head(5).itertuples():
        expected = float(np.abs(
            m.res.params[row.feature] * (X[row.feature] - X[row.feature].mean())
        ).mean())
        # the table rounds to 5dp, so compare at that precision
        assert abs(row.mean_abs_shap - expected) < 1e-5


def test_collinearity_splits_tree_attributions_too(fitted):
    """Not just coefficients: given both area columns, the GBM divides the size
    signal between them so neither looks as important as size actually is."""
    from rent_price_explainer.explain import shap_report
    ms, _, test = fitted
    attr = shap_report.compare_attributions(ms["inter"], ms["gbm"], test)
    twins = attr[attr["feature"].isin(["builtup_area", "carpet_area"])]
    assert len(twins) == 2, "the collinear pair should both be in the GBM matrix"
    combined = twins["gbm_mean_abs_shap"].sum()
    biggest = twins["gbm_mean_abs_shap"].max()
    assert biggest < combined, "the signal was not split at all"

    ols_size = attr.loc[attr["feature"] == "log_builtup_area",
                        "ols_mean_abs_shap"].iloc[0]
    assert ols_size > biggest, (
        "the single combined term should show size as more important than "
        "either half of the split pair")
