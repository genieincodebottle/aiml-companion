import dataclasses

import pytest

from cv_traps.config import CFG
from cv_traps.data.generate import generate
from cv_traps.evaluation.truth import split_panel


@pytest.fixture(scope="session")
def cfg():
    """A smaller config. Never points at the project's real data/ directory.

    Tests that write into the paths the documented commands read would poison
    the cache for anyone who runs pytest before running the pipeline.
    """
    return dataclasses.replace(CFG, n_customers=700, n_noise=40).validate()


@pytest.fixture(scope="session")
def panel(cfg):
    return generate(n_customers=cfg.n_customers, n_periods=cfg.n_periods,
                    n_noise=cfg.n_noise, seed=cfg.seed,
                    group_effect=cfg.group_effect, drift=cfg.drift,
                    missing_rate=cfg.missing_rate,
                    label_noise=cfg.label_noise)


@pytest.fixture(scope="session")
def exchangeable_panel(cfg):
    """The control: no latent customer effect, no drift.

    Deliberately larger than the other fixtures. The claim being tested is that
    grouped and ungrouped folds AGREE here, and agreement is a statement about
    a difference being small, which needs enough rows that the difference is
    not merely buried in fold-to-fold noise.
    """
    return generate(n_customers=2200, n_periods=cfg.n_periods,
                    n_noise=cfg.n_noise, seed=cfg.seed,
                    group_effect=0.0, drift=0.0,
                    missing_rate=cfg.missing_rate, label_noise=cfg.label_noise)


@pytest.fixture(scope="session")
def split(panel, cfg):
    dev, out = split_panel(panel, cfg)
    return {"dev": dev, "out": out}
