"""Build the model, or a deliberately broken copy of it, and record what happened.

A variant is built in its own folder under artifacts/work. The shipped dbt
project is copied there, the variant's edits are applied to the copy, and dbt
runs against a fresh DuckDB file. The shipped models are never edited.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

from star_schema import variants as V
from star_schema.config import load_config, resolve
from star_schema.evaluation.answer_key import score_connection
from star_schema.pipelines import dbt_driver

LAYERS = ["stg_customer_changes", "stg_orders", "stg_order_lines", "stg_products",
          "dim_customer", "dim_product", "fct_order", "fct_order_line",
          "rpt_revenue_by_region_category_day", "rpt_shipping_by_region_day"]


def _read_models(project: Path) -> dict[str, str]:
    files = {}
    for path in (project / "models").rglob("*.sql"):
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        files[path.relative_to(project).as_posix()] = text
    return files


def build_variant(variant: dict | None, cfg: dict | None = None, run_tests: bool = True) -> dict:
    """Build one variant (None means the shipped, correct model). Returns a plain dict."""
    cfg = cfg or load_config()
    artifacts = resolve(cfg, "artifacts")
    name = variant["id"] if variant else "baseline"
    work = artifacts / "work" / name
    project = dbt_driver.copy_project(resolve(cfg, "dbt_project"), work / "project")
    if variant:
        patched = V.patch_files(_read_models(project), variant["ops"])
        for rel, text in patched.items():
            (project / rel).write_text(text, encoding="utf-8", newline="\n")

    db_path = work / "warehouse.duckdb"
    result = dbt_driver.build(project, work, db_path, resolve(cfg, "raw_dir"), run_tests=run_tests)
    out = {"id": name, "title": variant["title"] if variant else "Shipped model, unmodified",
           "trap": variant["trap"] if variant else "-", "run_ok": result.run_ok,
           "run_errors": result.run_errors, "tests": [vars(t) for t in result.tests],
           "score": None, "rows": {}}
    if result.run_ok:
        con = duckdb.connect(str(db_path), read_only=True)
        out["score"] = score_connection(con, cfg)
        out["rows"] = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in LAYERS}
        con.close()
    shutil.rmtree(work / "project", ignore_errors=True)
    return out


def build_many(ids: list[str | None], cfg: dict | None = None, run_tests: bool = True,
               workers: int = 4) -> list[dict]:
    """Build several variants side by side. Each has its own folder and DuckDB file."""
    cfg = cfg or load_config()
    todo = [V.by_id(i) if i else None for i in ids]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda v: build_variant(v, cfg, run_tests), todo))
