"""Run the lltDistillation notebook (learner defaults + USE_OUR_ADAPTER = True) in one Python process on the GCP T4 VM.

Executes every code cell of llt-distill-notebook.ipynb in order, in one namespace, as Jupyter would. The %pip install cell
is skipped (requirements.txt installs the same pins first). Each cell prints "CELL <name> start/end <seconds>", and the
notebook's own stdout tee writes notebook_stdout.txt in the output folder, next to distill_results.json.

  python run_nb.py --out outputs
Spot preemption: the whole job is about 50 GPU-min with no long single step, so after a restart run the same command
again; it starts from the beginning (the output folder is cleared of the previous partial stdout first).
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs")
args = ap.parse_args()
assert sys.version_info >= (3, 11), "Python 3.11 or newer is needed (the notebook pins need it)"
here = Path(__file__).resolve().parent
nb = json.load(open(here / "llt-distill-notebook.ipynb", encoding="utf-8"))
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "notebook_stdout.txt").unlink(missing_ok=True)
os.chdir(out)                                   # adapters/, distill_results.json and notebook_stdout.txt land here
ns = {"__name__": "__main__"}
t0 = time.time()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if src.lstrip().startswith("# Pinned Hugging Face stack"):
        print("skip install cell (requirements.txt)", flush=True)
        continue
    exec(compile(src, "cell%02d" % i, "exec"), ns)
    sys.stdout.flush()
print("JOB DONE in", round((time.time() - t0) / 60, 1), "min", flush=True)