"""Check that the built notebook's page cells equal cells/*.py, the files that ran end to end on Kaggle.

Usage: python check_cells.py   (after node build_notebook.mjs <module file>)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "lltInstructionData.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]).rstrip() for c in nb["cells"] if c["cell_type"] == "code"]
page = code[2:2 + 12]                       # after the runtime check and the install cell
files = [(HERE / "cells" / f).read_text(encoding="utf-8").rstrip() for f in sorted(p.name for p in (HERE / "cells").glob("c*.py"))]
assert len(page) == len(files) == 12, (len(page), len(files))
for k, (a, b) in enumerate(zip(page, files), 1):
    assert a == b, "cell %d differs from cells/%s" % (k, sorted(p.name for p in (HERE / "cells").glob("c*.py"))[k - 1])
head = (HERE / "notebook_head.py").read_text(encoding="utf-8")
assert code[0] in head and code[1] in head, "runtime check or install cell differs from notebook_head.py"
print("notebook cells equal cells/*.py and notebook_head.py:", len(code), "code cells")
