"""Assemble the lltRL page draft: JS constants from steps/*.py and challenge files, then the hand-written body.

  python export_page_code.py <body.mjs> <out.mjs>
Each code string must hold no backslash, no backtick and no dollar-brace (they sit in JS template literals).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
body, out = Path(sys.argv[1]), Path(sys.argv[2])


def check(name, s):
    for bad in ("\\", "`", "${"):
        assert bad not in s, (name, bad)
    assert all(ord(ch) < 128 for ch in s), (name, "non-ASCII")
    return s


consts = []
for i, path in enumerate(sorted((HERE / "steps").glob("cell*_step*.py")), start=1):
    src = check(path.name, path.read_text(encoding="utf-8").rstrip())
    consts.append("const CELL%d_CODE = `%s`;" % (i, src))
sol = (HERE / "challenge_solution.py").read_text(encoding="utf-8").rstrip()
starter = (HERE / "challenge_starter.py").read_text(encoding="utf-8").rstrip()
consts.append("const CHALLENGE_STARTER = `%s`;" % check("starter", starter))
consts.append("const CHALLENGE_SOLUTION = `%s`;" % check("solution", sol))
out.write_text(body.read_text(encoding="utf-8").replace("// @@CODE@@", "\n\n".join(consts)), encoding="utf-8")
print("wrote", out, len(consts), "constants")
