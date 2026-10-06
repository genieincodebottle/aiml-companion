"""The answer key never reaches the build path."""
import re
from pathlib import Path

import duckdb

from star_schema.config import ROOT

# The only files allowed to name the key. The generator creates it, the writer
# saves it, the scorer reads it back, and the standalone notebook builder inlines
# the generator and scorer because the notebook generates and scores in one place.
ALLOWED = {
    "src/star_schema/data/world.py",
    "src/star_schema/data/io.py",
    "src/star_schema/evaluation/answer_key.py",
    "notebooks/_build_notebook.py",
}
KEY_SYMBOLS = re.compile(r"TRUE_(REVENUE|SHIPPING|LEDGER)|true_revenue|true_shipping|trap_ledger|key_dir")


def _files():
    for folder, patterns in (("src", ["*.py"]), ("warehouse", ["*.sql", "*.yml"]), ("conf", ["*.yaml"])):
        for pattern in patterns:
            yield from (ROOT / folder).rglob(pattern)
    yield ROOT / "run.py"
    yield ROOT / "notebooks" / "_build_notebook.py"


def test_no_module_outside_the_scorer_names_the_key():
    offenders = []
    for path in _files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED or rel == "conf/config.yaml":
            continue
        if KEY_SYMBOLS.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders, f"these files name the answer key: {offenders}"


def test_the_config_only_names_a_folder_for_the_key():
    text = (ROOT / "conf" / "config.yaml").read_text(encoding="utf-8")
    assert len(re.findall("key_dir", text)) == 1


def test_the_dbt_project_cannot_see_the_key(baseline, cfg):
    work = Path(cfg["paths"]["artifacts"]) / "work" / "baseline" / "warehouse.duckdb"
    con = duckdb.connect(str(work), read_only=True)
    tables = {r[0] for r in con.execute(
        "select table_schema || '.' || table_name from information_schema.tables").fetchall()}
    con.close()
    assert not any("true" in t.lower() or "key" in t.lower() for t in tables), tables
    assert {t for t in tables if t.startswith("raw.")} == {
        f"raw.{n}" for n in ("customer_changes", "orders", "order_lines", "products", "product_corrections")}


def test_dbt_models_use_only_ref_source_and_config():
    allowed = re.compile(r"^\{\{\s*(ref\('[a-z_]+'\)|source\('[a-z_]+',\s*'[a-z_]+'\)|config\([^)]*\))\s*\}\}$")
    for path in list((ROOT / "warehouse" / "models").rglob("*.sql")) + \
            list((ROOT / "warehouse" / "tests").rglob("*.sql")):
        if "generic" in path.parts:
            continue
        for call in re.findall(r"\{\{.*?\}\}|\{%.*?%\}", path.read_text(encoding="utf-8")):
            assert allowed.match(call), f"{path.name}: {call}"


def test_no_dbt_packages_are_needed():
    assert not (ROOT / "warehouse" / "packages.yml").exists()
    assert not (ROOT / "warehouse" / "dependencies.yml").exists()
