"""Shared fixtures. Every test runs fully offline: no key, no network, no cost.

That is a property of the stub provider rather than of mocking. The tests
exercise the real judge, the real RART loop, the real serving loop and the real
drift monitor - only the model behind each role is replaced by the deterministic
rule engine. A suite built on mocks of those components would pass whether or
not the components worked.
"""

from __future__ import annotations

import pytest

from src.config import as_offline, load_config
from src.domain import load_domain
from src.runtime import Runtime
from src.services._context import build_context


@pytest.fixture(scope="session")
def config():
    return as_offline(load_config())


@pytest.fixture(scope="session")
def domain(config):
    return load_domain(config.domain_name)


@pytest.fixture()
def runtime(config):
    return Runtime(config)


@pytest.fixture()
def context():
    return build_context(offline=True)
