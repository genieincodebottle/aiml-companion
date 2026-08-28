"""The generator must actually contain the violations the project teaches.
If a violation quietly disappears, every downstream lesson becomes a lie."""
import numpy as np
import pytest

from rent_price_explainer.data import schema
from rent_price_explainer.data.generate import TRUE_BETAS, generate


def test_schema_valid(listings):
    assert schema.validate(listings) is listings


def test_rents_are_plausible(listings):
    med = listings["monthly_rent"].median()
    assert 8_000 < med < 80_000, f"median rent {med:,.0f} is not a plausible rent"


def test_collinearity_is_actually_present(listings):
    """The whole VIF lesson depends on this pair being near-duplicates."""
    r = listings["carpet_area"].corr(listings["builtup_area"])
    assert r > 0.95, f"planted collinearity has vanished (r={r:.3f})"


def test_heteroscedasticity_is_present(listings):
    """Residual spread must grow with size, or the Breusch-Pagan lesson dies.

    Measured on residuals from the FULL specification -- a one-variable fit
    leaves so much omitted-variable variance that the planted effect is
    invisible, which is itself a decent illustration of why you diagnose on
    the model you actually fitted.
    """
    from rent_price_explainer.models.linear import SpecifiedOLS

    m = SpecifiedOLS().fit(listings)
    resid = m.res.resid
    area = listings["builtup_area"].to_numpy()
    q1, q4 = np.quantile(area, 0.25), np.quantile(area, 0.75)
    small = resid[area <= q1].std()
    large = resid[area >= q4].std()
    assert large > small * 1.5, (
        f"planted heteroscedasticity is gone: sd {small:.3f} -> {large:.3f}")


def test_age_effect_is_non_monotone(listings):
    """A U-shape, not a line -- otherwise the RESET lesson is trivial."""
    b = listings.groupby(pd.cut(listings["age_years"], 6),
                         observed=True)["monthly_rent"].median()
    diffs = np.sign(np.diff(b.values))
    assert len(set(diffs)) > 1, "age effect is monotone; the U-shape is gone"


def test_junk_features_are_pure_noise(listings):
    for c in [c for c in listings.columns if c.startswith("junk_")]:
        r = abs(listings[c].corr(listings["monthly_rent"]))
        assert r < 0.10, f"{c} correlates with rent at {r:.3f} -- not junk"


def test_generator_is_deterministic():
    a, b = generate(n=300, seed=7), generate(n=300, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_answer_key_is_not_imported_by_fitting_code():
    """TRUE_BETAS must reach only the recovery scorer, never a model."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "rent_price_explainer"
    offenders = []
    for f in root.rglob("*.py"):
        if f.parts[-2:] in [("evaluation", "recovery.py")] or f.name == "generate.py":
            continue
        if "TRUE_BETAS" in f.read_text(encoding="utf-8"):
            offenders.append(str(f.relative_to(root)))
    assert not offenders, f"answer key leaked into fitting code: {offenders}"


import pandas as pd  # noqa: E402  (imported late for the cut() helper above)
