# build_module.py - assemble lltMemoryThroughput.mjs from the hand-written parts and the step files.
# Code blocks come from steps/*.py and challenges/*.py verbatim, so the page shows exactly what ran.
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "lltMemoryThroughput.mjs")
FILES = {
    "check": "steps/check_runtime.py", "gpt": "steps/gpt.py", "s01": "steps/s01_static.py", "s02": "steps/s02_predict.py",
    "s03": "steps/s03_grid.py", "s04": "steps/s04_timing.py", "s05": "steps/s05_cut.py", "s06": "steps/s06_precision.py",
    "allreduce": "steps/allreduce.py", "ddp": "steps/ddp.py", "fsdp": "steps/fsdp.py", "fsdp_ckpt": "steps/fsdp_ckpt.py",
    "tiny": "steps/tiny.py", "sweep": "steps/sweep.py", "prep": "steps/prep.py",
    "launch_ddp": "steps/launch_ddp.sh", "launch_p3": "steps/launch_p3.sh", "read_sweep": "steps/read_sweep.py",
    "ch1_starter": "challenges/peak_starter.py", "ch1_solution": "challenges/peak_solution.py",
    "ch2_starter": "challenges/zero_starter.py", "ch2_solution": "challenges/zero_solution.py",
}
lines = ["const CODE = {"]
for key, rel in FILES.items():
    p = os.path.join(LAB, rel)
    if not os.path.exists(p):
        print("missing", rel); continue
    t = open(p, encoding="utf-8").read().rstrip()
    for bad in ("`", "\\", "${"):
        assert bad not in t, f"{rel} contains {bad!r}"
    lines.append(f"    {key}: `{t}`,")
lines.append("};\n")
body = open(os.path.join(HERE, "head.mjs"), encoding="utf-8").read()
for part in ("body_part1.mjs", "body_part2.mjs", "body_part3.mjs"):
    p = os.path.join(HERE, part)
    body = body.replace(f"/*@@{part}@@*/", open(p, encoding="utf-8").read() if os.path.exists(p) else "")
tail = open(os.path.join(HERE, "tail.mjs"), encoding="utf-8").read()
tail = tail.replace("        @@QUIZ@@" + chr(10), open(os.path.join(HERE, "quiz.mjs"), encoding="utf-8").read())
tail = tail.replace("        @@IQA@@" + chr(10), open(os.path.join(HERE, "iqa.mjs"), encoding="utf-8").read())
body += tail
for key, val in (json.load(open(os.path.join(HERE, "fill.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "fill.json")) else {}).items():
    body = body.replace("@@" + key + "@@", val)
src = body.replace("/*@@CODE@@*/", "\n".join(lines))
left = re.findall(r"@@[A-Z0-9_]+@@", src)
if left and not os.environ.get("ALLOW_PLACEHOLDERS"):
    raise SystemExit(f"unfilled placeholders {sorted(set(left))}")
print("placeholders left:", sorted(set(left)))
for bad in ("—", "–", "“", "”", "‘", "’"):
    assert bad not in src, f"banned character {bad!r}"
open(OUT, "w", encoding="utf-8").write(src)
print("wrote", OUT, len(src), "chars")
