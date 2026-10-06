"""Build the dbt project from Python.

Each build loads the raw CSVs into a fresh DuckDB file, then runs a worker
process that calls dbt through its programmatic `dbtRunner`. --project-dir and
--profiles-dir are passed explicitly, so nothing reads or writes ~/.dbt, and
build artefacts (target, logs) land beside the DuckDB file, never in the
shipped project folder.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from star_schema.config import ROOT

RAW_TABLES = ["customer_changes", "orders", "order_lines", "products", "product_corrections"]
IGNORE = shutil.ignore_patterns("target", "logs", "dbt_packages", "__pycache__")


@dataclass
class TestOutcome:
    unique_id: str
    label: str
    tier: str            # generic, singular or added
    status: str          # pass, fail or error
    failures: int


@dataclass
class DbtResult:
    run_ok: bool
    run_errors: list[str] = field(default_factory=list)
    tests: list[TestOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[TestOutcome]:
        return [t for t in self.tests if t.status != "pass"]


def load_raw(db_path: Path, raw_dir: Path) -> None:
    """Load every raw CSV into schema raw as text. Staging owns all casting."""
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists raw")
    for table in RAW_TABLES:
        csv_path = (raw_dir / f"{table}.csv").as_posix()
        con.execute(f"create or replace table raw.{table} as "
                    f"select * from read_csv('{csv_path}', header = true, all_varchar = true)")
    con.close()


def copy_project(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=IGNORE)
    return dest


def build(project: Path, work: Path, db_path: Path, raw_dir: Path,
          run_tests: bool = True) -> DbtResult:
    """Fresh DuckDB file, load raw, `dbt run`, then `dbt test`.

    run + test rather than `dbt build`, because `dbt build` skips everything
    downstream of a failed test and the matrix needs every test to get its say."""
    work.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    result_file = work / "dbt_result.json"
    if result_file.exists():
        result_file.unlink()
    load_raw(db_path, raw_dir)

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "star_schema.pipelines.dbt_worker", str(project), str(work),
         str(db_path), "1" if run_tests else "0"],
        env=env, capture_output=True, text=True, encoding="utf-8")
    if not result_file.exists():
        return DbtResult(False, [f"dbt worker crashed: {proc.stderr.strip()[-600:]}"])
    data = json.loads(result_file.read_text(encoding="utf-8"))
    tests = [TestOutcome(**t) for t in data["tests"]]
    tests.sort(key=lambda t: (t.tier, t.label))
    return DbtResult(data["run_ok"], data["run_errors"], tests)
