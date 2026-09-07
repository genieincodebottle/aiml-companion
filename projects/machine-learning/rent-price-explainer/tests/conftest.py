import pytest
from sklearn.model_selection import train_test_split

from rent_price_explainer.data.generate import generate


@pytest.fixture(scope="session")
def listings():
    return generate(n=1500, seed=42)


@pytest.fixture(scope="session")
def splits(listings):
    return train_test_split(listings, test_size=0.25, random_state=42)
