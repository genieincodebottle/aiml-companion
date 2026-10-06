"""The answer key. The only module outside the generator that may read it.

The key is written next to the raw data by the generator and read back here to
score a built report. Nothing in the dbt project, the staging models, the
marts or the variant builders imports this module's data, and
tests/test_lessons.py scans the source tree to prove it.
"""
from __future__ import annotations

import json
from pathlib import Path

from star_schema.config import load_config, resolve
from star_schema.evaluation.scoring import score_revenue, score_shipping

REVENUE_FILE = "true_revenue.json"
SHIPPING_FILE = "true_shipping.json"
LEDGER_FILE = "trap_ledger.json"


def write_key(key_dir: Path, world: dict) -> None:
    key_dir.mkdir(parents=True, exist_ok=True)
    for name, field in ((REVENUE_FILE, "TRUE_REVENUE"), (SHIPPING_FILE, "TRUE_SHIPPING"),
                        (LEDGER_FILE, "TRUE_LEDGER")):
        text = json.dumps(world[field], sort_keys=True, indent=1) + "\n"
        (key_dir / name).write_text(text, encoding="utf-8", newline="\n")


def load_key(cfg: dict | None = None, key_dir: Path | None = None) -> tuple[list, list, dict]:
    key_dir = key_dir or resolve(cfg or load_config(), "key_dir")
    return tuple(json.loads((key_dir / f).read_text(encoding="utf-8"))
                 for f in (REVENUE_FILE, SHIPPING_FILE, LEDGER_FILE))


def report_rows(con) -> tuple[list, list]:
    """Read both report tables as text, so a float column shows its real digits."""
    revenue = con.execute(
        "select cast(order_date as varchar), region, category, units, order_lines, "
        "cast(line_revenue as varchar) from rpt_revenue_by_region_category_day").fetchall()
    shipping = con.execute(
        "select cast(order_date as varchar), region, orders, "
        "cast(shipping_revenue as varchar) from rpt_shipping_by_region_day").fetchall()
    return revenue, shipping


def score_connection(con, cfg: dict | None = None, key_dir: Path | None = None) -> dict:
    """Score the report tables in an open DuckDB connection against the key."""
    true_revenue, true_shipping, _ = load_key(cfg, key_dir)
    revenue, shipping = report_rows(con)
    return {"revenue": score_revenue(revenue, true_revenue),
            "shipping": score_shipping(shipping, true_shipping)}


def showcase_orders(cfg: dict | None = None, key_dir: Path | None = None) -> list[dict]:
    return load_key(cfg, key_dir)[2]["showcase_orders"]
