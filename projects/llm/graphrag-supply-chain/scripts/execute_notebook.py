#!/usr/bin/env python
"""Execute the notebook in place so it ships with its outputs saved.

    python scripts/execute_notebook.py

WHY. A notebook with no saved outputs shows a reader nothing until they have a
database, a key and ten minutes. Most people open it on GitHub to decide whether
the project is worth their evening, and an empty notebook answers no.

It runs from the project root, which is what the notebook's own first cell
assumes, and it needs the same things `run.py doctor` needs: a Gemini key and a
populated graph. Outputs are written back into the same file, so re-running it
after a code change refreshes what a reader sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "GraphRAG_Supply_Chain.ipynb"


def main() -> int:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("nbformat and nbclient are needed to execute the notebook:")
        print("    pip install nbformat nbclient ipykernel")
        return 1

    if not NB.exists():
        print(f"notebook not found: {NB}")
        return 1

    nb = nbformat.read(NB, as_version=4)
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"executing {code_cells} code cells from {NB.name}")
    print("this calls the model and reads the graph, so `run.py doctor` should be green first\n")

    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()

    # Execution counts are noise in a diff and change on every run even when
    # nothing else does. Clearing them keeps the file reviewable.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell["execution_count"] = None
            for out in cell.get("outputs", []):
                out.pop("execution_count", None)

    nbformat.write(nb, NB)

    with_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"\nwrote {NB.name}: {with_out} of {code_cells} code cells now carry output")
    if with_out < code_cells - 2:
        print("WARNING: more cells than expected produced nothing. Check for silent no-ops.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
