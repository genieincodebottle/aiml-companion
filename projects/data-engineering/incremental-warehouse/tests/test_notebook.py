"""The executed notebook and the repo agree on every number."""
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "incremental_warehouse_standalone.ipynb"


def _saved_results():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", ""))
            if "NOTEBOOK_RESULTS " in text:
                return json.loads(text.split("NOTEBOOK_RESULTS ", 1)[1])
    pytest.fail("the notebook has no saved NOTEBOOK_RESULTS output, execute it first")


def test_notebook_numbers_equal_the_repo_numbers(results, clean_memory):
    saved = _saved_results()
    repo = json.loads(json.dumps(results, default=str))
    assert saved["naive"] == repo
    assert saved["clean_checksums"] == clean_memory.wh.checksums()


def test_notebook_runs_the_same_sql_as_the_repo():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for sql in sorted((NOTEBOOK.parents[1] / "sql").glob("*.sql")):
        assert sql.read_text(encoding="utf-8").rstrip() in code, f"{sql.name} drifted"


def test_notebook_has_no_errors_and_imports_nothing_from_the_package():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "from warehouse" not in code and "import warehouse" not in code
    errors = [o for c in nb["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert errors == []
    assert sum(1 for c in nb["cells"] for o in c.get("outputs", [])
               if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})) == 7
