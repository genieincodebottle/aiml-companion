"""Runs `dbt run` and `dbt test` once, in its own process, and writes a JSON result.

WHY a separate process: dbt-duckdb holds the DuckDB file open for the life of
the process, so the parent could not read the finished warehouse. A process per
build also means one build can never leak state into the next, and builds of
different variants can run side by side.

    python -m star_schema.pipelines.dbt_worker PROJECT WORK DB RUN_TESTS
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _test_label(node) -> tuple[str, str]:
    """A short readable name and the tier of a dbt test node."""
    tier_added = "added" in node.tags
    meta = getattr(node, "test_metadata", None)
    if meta is None:
        return node.name, "added" if tier_added else "singular"
    kwargs = {k: v for k, v in meta.kwargs.items() if k not in ("model", "column_name")}
    column = meta.kwargs.get("column_name")
    model = (getattr(node, "attached_node", "") or "").split(".")[-1]
    target = f"{model}.{column}" if column else model
    extra = ""
    if "combination_of_columns" in kwargs:
        extra = "(" + ",".join(kwargs["combination_of_columns"]) + ")"
    elif "to" in kwargs:
        extra = "->" + str(kwargs["to"]).replace("ref('", "").replace("')", "")
    return f"{meta.name}[{target}]{extra}", "added" if tier_added else "generic"


def main(project: str, work: str, db: str, run_tests: str) -> int:
    from dbt.cli.main import dbtRunner

    os.environ["STAR_SCHEMA_DB"] = db
    os.environ["DBT_SEND_ANONYMOUS_USAGE_STATS"] = "False"
    work_dir = Path(work)
    base = ["--project-dir", project, "--profiles-dir", project,
            "--target-path", str(work_dir / "target"), "--log-path", str(work_dir / "logs"),
            "--quiet"]
    out: dict = {"run_ok": True, "run_errors": [], "tests": []}

    run = dbtRunner().invoke(["run"] + base)
    if not run.success:
        out["run_ok"] = False
        if run.exception:
            out["run_errors"].append(str(run.exception))
        for r in (run.result.results if run.result else []):
            if str(r.status) != "success":
                out["run_errors"].append(f"{r.node.name}: {r.message}")
    elif run_tests == "1":
        test = dbtRunner().invoke(["test"] + base)
        for r in (test.result.results if test.result else []):
            label, tier = _test_label(r.node)
            out["tests"].append({"unique_id": r.node.unique_id, "label": label, "tier": tier,
                                 "status": str(r.status), "failures": int(r.failures or 0)})
    (work_dir / "dbt_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
