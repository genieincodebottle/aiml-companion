import dataclasses

import pytest

from support_ticket_triage.config import CFG
from support_ticket_triage.data.generate import generate
from support_ticket_triage.features.build import split, xy


@pytest.fixture(scope="session")
def cfg():
    """A small config. Never points at the project's real data/ directory.

    Tests that write into the paths the documented commands read would poison
    the cache for anyone who runs pytest before running the pipeline.
    """
    return dataclasses.replace(CFG, n_tickets=2500).validate()


@pytest.fixture(scope="session")
def tickets(cfg):
    return generate(n_tickets=cfg.n_tickets, seed=cfg.seed,
                    dependency_strength=cfg.dependency_strength)


@pytest.fixture(scope="session")
def independent_tickets(cfg):
    """The same inbox with the assumption switched ON, as a control.

    Deliberately larger than the other fixtures. These are per-CLASS estimates,
    and the rarest class is 3% of the data, so at 2,500 tickets it has ~75 rows
    and a lift estimated from it swings between 0.65 and 1.22 on noise alone.
    Measuring a distributional property needs enough rows in the smallest cell,
    not enough rows overall.
    """
    return generate(n_tickets=20000, seed=cfg.seed, dependency_strength=0.0)


@pytest.fixture(scope="session")
def fitted(tickets, cfg):
    train, test = split(tickets, cfg)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "train": train, "test": test}
