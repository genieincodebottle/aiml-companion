"""Run the notebook's research cell (steps/nb_research.py) after steps/nb_model.py and record its output.

    python run_research_cell.py --threads 2
"""
import argparse
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()
import torch  # noqa: E402
torch.set_num_threads(args.threads)
ns = {"__name__": "__main__"}
out = {}
for name in ("nb_model.py", "nb_research.py"):
    path = os.path.join(HERE, "steps", name)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    out[name] = buf.getvalue()
    print(out[name], end="", flush=True)
from runlog import write_run  # noqa: E402
write_run("lltGeneration-t1-notebook-research-cell", "T1", "research:induction-toy-notebook", {"threads": args.threads}, 0, [],
          evals={"stdout": out["nb_research.py"]})
