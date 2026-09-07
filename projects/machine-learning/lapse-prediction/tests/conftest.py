import pandas as pd
import pytest

from lapse_prediction.config import CFG
from lapse_prediction.data.generate import generate
from lapse_prediction.features.build import build
from lapse_prediction.features.labels import add_labels, mature


@pytest.fixture(scope="session")
def ledger() -> pd.DataFrame:
    """Small but structurally complete synthetic book."""
    return generate(n_policies=400, seed=CFG.seed)


@pytest.fixture(scope="session")
def modelling_table(ledger) -> pd.DataFrame:
    return mature(add_labels(build(ledger)))
