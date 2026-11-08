"""Check that the notebook's page cells equal cells/*.py (the code that ran on Kaggle) and the head cells equal notebook_head.py.
Usage: python check_cells.py
"""
import json, runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = json.load(open(HERE / "lltEvaluation.ipynb", encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
head = runpy.run_path(str(HERE / "notebook_head.py"))
cells = [p.read_text(encoding="utf-8").rstrip() for p in sorted((HERE / "cells").glob("c*.py"))]
challenge = (HERE / "browser" / "challenge_starter.py").read_text(encoding="utf-8").rstrip()
want = [head["RUNTIME_CHECK"], head["INSTALL"]] + cells + [challenge]
assert len(code) == len(want), (len(code), len(want))
for i, (a, b) in enumerate(zip(code, want)):
    assert a.rstrip() == b.rstrip(), "code cell " + str(i + 1) + " differs"
print("notebook matches", len(want), "cells")
