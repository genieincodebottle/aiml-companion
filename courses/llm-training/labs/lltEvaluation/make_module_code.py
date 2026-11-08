"""Write the page's code constants into the module draft from cells/*.py and browser/*.py.

The cells are the single source: the notebook, the Kaggle kernels and the page all read them.
Usage: python make_module_code.py <path to lltEvaluation.mjs>
Rewrites the block between '// BEGIN GENERATED CODE' and '// END GENERATED CODE'.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONSTS = [(p.stem.split("_", 1)[0].upper() + "_" + p.stem.split("_", 1)[1].upper(), "cells/" + p.name)
          for p in sorted((HERE / "cells").glob("c*.py"))]
CONSTS += [("CHALLENGE_STARTER", "browser/challenge_starter.py"), ("CHALLENGE_SOLUTION", "browser/challenge_solution.py")]

out = ["// BEGIN GENERATED CODE (make_module_code.py from labs/lltEvaluation/cells and browser; do not edit by hand)"]
for name, rel in CONSTS:
    src = (HERE / rel).read_text(encoding="utf-8").rstrip()
    for bad in ["\\", "`", "${"]:
        assert bad not in src, (rel, bad)
    out.append("const " + name + " = `" + src + "`;")
    out.append("")
out.append("// END GENERATED CODE")
block = "\n".join(out)

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text, n = re.subn(r"// BEGIN GENERATED CODE.*?// END GENERATED CODE", lambda m: block, text, flags=re.S)
assert n == 1, "markers not found"
path.write_text(text, encoding="utf-8")
print("wrote", len(CONSTS), "code constants into", path)
