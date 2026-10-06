"""Shared fixtures. Everything is built once per session into a temp folder, so
the suite never depends on, or overwrites, the data/ and artifacts/ folders."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from star_schema.config import ROOT, load_config
from star_schema.data.io import write_world
from star_schema.evaluation.answer_key import load_key
from star_schema.pipelines import matrix as M
from star_schema.pipelines import report as R


def hash_tree(folder: Path) -> str:
    """One digest over every file under folder, in a fixed order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def temp_config(base: Path) -> dict:
    cfg = load_config()
    cfg["paths"]["raw_dir"] = str(base / "raw")
    cfg["paths"]["key_dir"] = str(base / "key")
    cfg["paths"]["artifacts"] = str(base / "artifacts")
    return cfg


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> dict:
    return temp_config(tmp_path_factory.mktemp("run1"))


@pytest.fixture(scope="session")
def raw_hashes(cfg) -> dict:
    return write_world(cfg)


@pytest.fixture(scope="session")
def key(cfg, raw_hashes):
    return load_key(cfg)


@pytest.fixture(scope="session")
def ledger(key) -> dict:
    return key[2]


@pytest.fixture(scope="session")
def project_hash_before_and_after(cfg, raw_hashes):
    """Run every variant and report whether the shipped dbt project changed."""
    project = ROOT / cfg["paths"]["dbt_project"]
    before = hash_tree(project)
    results = M.build_many(R.variant_ids(), cfg, run_tests=True, workers=4)
    return results, before, hash_tree(project)


@pytest.fixture(scope="session")
def results(project_hash_before_and_after) -> dict:
    """Every build result, keyed by variant id. 'baseline' is the correct model."""
    return {r["id"]: r for r in project_hash_before_and_after[0]}


@pytest.fixture(scope="session")
def baseline(results) -> dict:
    return results["baseline"]
