import pytest

from warehouse import naive
from warehouse.config import load_config
from warehouse.harness import Scenario
from warehouse.options import PipelineOptions
from warehouse.source.world import generate_world


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def world(cfg):
    return generate_world(cfg)


@pytest.fixture(scope="session")
def days(world):
    return world[0]


@pytest.fixture(scope="session")
def key(world):
    return world[1]


@pytest.fixture(scope="session")
def opts(cfg):
    return PipelineOptions.from_cfg(cfg)


@pytest.fixture(scope="session")
def clean(cfg, days, opts, tmp_path_factory):
    """One full correct pass on disk (Parquet raw store, DuckDB file)."""
    sc = Scenario(cfg, days, opts, root=tmp_path_factory.mktemp("clean") / "run")
    sc.runs = [sc.run_day(d) for d in range(1, cfg["source"]["days"] + 1)]
    return sc


@pytest.fixture(scope="session")
def clean_memory(cfg, days, opts):
    """The same pass with the in-memory store, which is what the notebook uses."""
    sc = Scenario(cfg, days, opts)
    sc.runs = [sc.run_day(d) for d in range(1, cfg["source"]["days"] + 1)]
    return sc


@pytest.fixture(scope="session")
def results(cfg, days, key, clean_memory):
    """Every naive-against-correct comparison, measured once for the whole session."""
    return naive.run_all(cfg, days, key, clean_memory)
