"""The diagnostics must catch what we planted, and clear what we fixed."""
import numpy as np
import pytest

from rent_price_explainer.diagnostics import assumptions
from rent_price_explainer.models.linear import InteractionOLS, NaiveOLS, SpecifiedOLS


@pytest.fixture(scope="module")
def naive(splits):
    train, _ = splits
    return NaiveOLS().fit(train), train


def test_naive_model_fails_multiple_checks(naive):
    m, train = naive
    d = assumptions.run_all(m.res, m._prepare(train))
    assert d["n_failed"] >= 4, "the naive model should fail most checks"


def test_vif_flags_the_planted_twin(naive):
    m, train = naive
    check, table = assumptions.vif(m._prepare(train))
    assert not check.passed
    top2 = set(table.head(2)["feature"])
    assert {"builtup_area", "carpet_area"} & top2, \
        f"VIF did not surface the collinear pair; top was {top2}"
    assert check.statistic > 20, "the planted collinearity should be extreme"


def test_reset_catches_the_wrong_functional_form(naive):
    m, train = naive
    assert not assumptions.reset_test(m.res).passed


def test_breusch_pagan_catches_the_planted_heteroscedasticity(naive):
    m, train = naive
    X = m._prepare(train)
    assert not assumptions.breusch_pagan(np.asarray(m.res.resid), X).passed


def test_fixes_improve_the_diagnostics(splits):
    """Not 'all green' -- honest improvement. With thousands of rows these
    tests reject trivial deviations, so the statistic falling matters more
    than the pass/fail flipping."""
    train, _ = splits
    naive_m = NaiveOLS().fit(train)
    spec_m = SpecifiedOLS().fit(train)

    n = assumptions.run_all(naive_m.res, naive_m._prepare(train))
    s = assumptions.run_all(spec_m.res, spec_m._prepare(train))
    assert s["n_failed"] < n["n_failed"], "the fixes did not reduce failures"

    n_vif = {c.name: c for c in n["checks"]}["multicollinearity"].statistic
    s_vif = {c.name: c for c in s["checks"]}["multicollinearity"].statistic
    assert s_vif < n_vif / 5, f"VIF barely moved: {n_vif:.1f} -> {s_vif:.1f}"


def test_every_check_states_a_consequence():
    """A diagnostic that does not say what the failure COSTS is decoration."""
    train = None
    from rent_price_explainer.data.generate import generate
    m = NaiveOLS().fit(generate(n=500, seed=1))
    d = assumptions.run_all(m.res, m._prepare(generate(n=500, seed=1)))
    for c in d["checks"]:
        assert len(c.consequence) > 25, f"{c.name} has no stated consequence"
