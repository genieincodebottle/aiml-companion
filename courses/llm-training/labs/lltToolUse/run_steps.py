"""Run the page's CPU cells (steps/cellNN_stepK.py) in order in one namespace, the way the notebook does.

Usage: python -I run_steps.py <bfcl_eval parent dir or ''> <cell numbers, e.g. 1 2 3 5 6 7>
bfcl_eval is pip-installed --no-deps in the notebook; locally pass the folder that holds an unpacked bfcl_eval package.
Prints each cell's output and its wall time, and writes run_steps_out.json with both.
"""
import contextlib, io, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
if sys.argv[1]:
    sys.path.insert(0, sys.argv[1])
import torch
torch.set_num_threads(4)
want = [int(x) for x in sys.argv[2:]]
ns = {"__name__": "__main__"}
out = {}
for f in sorted(os.listdir(os.path.join(HERE, "steps"))):
    if not f.startswith("cell"):
        continue
    n = int(f[4:6])
    if want and n not in want:
        continue
    code = open(os.path.join(HERE, "steps", f), encoding="utf-8").read()
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, f, "exec"), ns)
    sec = round(time.time() - t0, 1)
    out[f] = {"sec": sec, "stdout": buf.getvalue()}
    print("=====", f, sec, "s")
    print(buf.getvalue())
json.dump(out, open(os.path.join(HERE, "run_steps_out.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
