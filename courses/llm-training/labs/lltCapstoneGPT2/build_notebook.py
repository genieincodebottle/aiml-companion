"""Build the learner notebook lltCapstoneGPT2.ipynb (capstone 1, pretrain a 124M GPT). Run: python build_notebook.py

The commands in the cells are the ones on the capstone page (Setup tab and the "quick, full and compute-optimal paths"
section). PATH picks which of the three runs; the default is the quick path, which downloads our step 4,529 checkpoint
from the Hub, trains the last 240 steps, scores the model and GPT-2 in lm-evaluation-harness, and accounts the compute.
The "How to run this notebook" help cell comes from ../../tools/add_run_help.py, so every lab shows the same one.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

_src = (HERE.parent.parent / "tools" / "add_run_help.py").read_text(encoding="utf-8")
_ns = {}
exec("MARK = " + _src.split("MARK = ", 1)[1].split("\nroot =", 1)[0], _ns)
HELP = _ns["TEXT"]
assert _ns["MARK"] in HELP[0]

INTRO = """# Capstone: pretrain your own GPT-2 (quick, full and compute-optimal paths)

Pretrain a 124M-parameter GPT on FineWeb-Edu, score it against `openai-community/gpt2` in lm-evaluation-harness, and account for the FLOPs and the GPU-hours. LLM Training Lab, capstone 1.

Pick the path in the settings cell:

- **quick** (the default): download our full training state at step 4,529, train the last 240 steps of the 2.5B-token decay on text the checkpoint has not seen, then score and account.
- **full**: train 124M from random weights for 1B tokens (1,908 steps) in one session.
- **optimal**: train 124M from random weights for 2.5B tokens (4,769 steps) over two sessions, with an exact resume.

Use two T4 GPUs (Kaggle: Accelerator *GPU T4 x2*). One T4 also works, at about half the speed: the script reads the GPU count and keeps the batch at 524,288 tokens per step. A T4 has no bf16, so training runs in fp16 with a GradScaler."""

RUNTIME = '''import os, sys, platform, shutil, subprocess
problems = []
if sys.version_info < (3, 10):
    problems.append("Python 3.10 or newer is needed. Start a current Colab or Kaggle session.")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:                                   # a CPU runtime has no nvidia-smi
    smi = []
NPROC = len(smi)
if not smi:
    problems.append("No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
elif any("P100" in g for g in smi):
    problems.append("A P100 has no kernels in current PyTorch. Pick a T4.")
elif NPROC not in (1, 2, 4):
    problems.append("Use 1, 2 or 4 GPUs: the 524,288-token step has to split evenly (micro-batch 8 x 1,024 tokens x GPUs x accumulation steps).")
free_disk = shutil.disk_usage(".").free / 1e9
if free_disk < 20:
    problems.append("Free at least 20 GB of disk for the data, the checkpoint and the model downloads (" + str(round(free_disk)) + " GB free).")
print("Python", platform.python_version(), "| GPUs:", "; ".join(smi) or "none", "| disk free", round(free_disk), "GB")
for p in problems:
    print("PROBLEM", p)
print("Runtime OK" if not problems else "Read the PROBLEM lines first.")'''

INSTALL = '''# Pinned versions of our recorded runs. torch and numpy stay as the runtime ships them.
%pip install -q tiktoken==0.14.0 "lm-eval[hf]==0.4.13" pyarrow huggingface_hub
import importlib.metadata as md
print({p: md.version(p) for p in ("torch", "transformers", "lm_eval", "tiktoken", "huggingface_hub", "numpy")})'''

SETTINGS = '''import time, json, glob
PATH = "quick"          # "quick" | "full" | "optimal"
STOP_AFTER_SEC = 36000 if PATH == "optimal" else 0   # the optimal path stops cleanly after 10 h and saves a checkpoint
RESUME_FROM = ""        # optimal path, session 2: the checkpoint folder from session 1, e.g. /kaggle/input/<session-1-output>/run2p5b/ckpt/step_00XXXX
CFG = {   # data files, run folder, schedule: the same flags as the page
    "quick":   dict(files="0,12,13 --max-docs 150000", out="decay",   run_id="my-124m-quick", start_step=4529,
                    sched="--stable-end 4529 --max-steps 4769 --reset-loader-shard 1"),
    "full":    dict(files="0,1",                       out="run1b",   run_id="my-124m-1B",    start_step=0,
                    sched="--stable-end 1813 --max-steps 1908 --milestones 1908 --export-every 500"),
    "optimal": dict(files="0,1,2,3",                   out="run2p5b", run_id="my-124m-2p5B",  start_step=0,
                    sched="--stable-end 4529 --max-steps 4769 --milestones 4529,4769"),
}[PATH]
DATA, OUT, RUN_ID, START_STEP = "/tmp/data", CFG["out"], CFG["run_id"], CFG["start_step"]
LAST_STEP = int(CFG["sched"].split("--max-steps ")[1].split()[0])
STAGES = {}             # seconds per stage, filled by the cells below
print(PATH, "path | data files", CFG["files"].split()[0], "| steps", START_STEP, "->", LAST_STEP, "| GPUs", NPROC)'''

GET_CODE = '''# Reuse the folder if the notebook already runs inside the lab, else clone just this lab
if not (os.path.exists("train.py") and os.path.exists("gpt.py")):
    WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content" if os.path.isdir("/content") else os.getcwd()
    os.chdir(WORK)
    if not os.path.isdir("aiml-companion"):
        !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/genieincodebottle/aiml-companion.git
        !cd aiml-companion && git sparse-checkout set courses/llm-training/labs/lltCapstoneGPT2
    os.chdir("aiml-companion/courses/llm-training/labs/lltCapstoneGPT2")
print(os.getcwd())'''

PREP = '''# FineWeb-Edu to GPT-2 tokens. Safe to rerun: it is skipped when the data is already there. /tmp keeps Kaggle from saving it as output.
t0 = time.time()
if not os.path.exists(DATA + "/index.json"):
    !python prep_fineweb.py --out {DATA} --file-ids {CFG["files"]}
STAGES["prep"] = time.time() - t0
assert os.path.exists(DATA + "/index.json"), "data prep did not finish: read the output above"
print("data prep", round(STAGES["prep"]), "s")'''

FETCH = '''# Quick path only: our step 4,529 training state (about 1.5 GB), checked against the repository's manifest.json
t0 = time.time()
if PATH == "quick" and not os.path.exists("step_004529_state.pt"):
    !python fetch_hub.py step_004529_state.pt
STAGES["download"] = time.time() - t0
assert PATH != "quick" or os.path.exists("step_004529_state.pt"), "the checkpoint download failed: read the output above"
print("download", round(STAGES["download"]), "s")'''

TRAIN = '''# Quick path: 240 steps from our checkpoint. Full and optimal: from random weights. After a stop, rerun the cell: it resumes from the newest checkpoint.
if glob.glob(OUT + "/ckpt/step_*[0-9]"):
    RESUME = "--resume auto"                       # our own checkpoint, same session or after a Kaggle or spot stop
elif RESUME_FROM:
    RESUME = "--resume " + RESUME_FROM
elif PATH == "quick":
    RESUME = "--resume step_004529_state.pt"
else:
    RESUME = ""
SCHED = CFG["sched"] if RESUME != "--resume auto" else CFG["sched"].replace("--reset-loader-shard 1", "")   # the data position is already in the checkpoint
SAVE = "--save-every 40 --keep-last 1" if PATH == "quick" else ""
STOP = "--stop-after-sec " + str(STOP_AFTER_SEC) if STOP_AFTER_SEC else ""
t0 = time.time()
!torchrun --nproc_per_node={NPROC} train.py --data {DATA} --out {OUT} --run-id {RUN_ID} --micro 8 {RESUME} {SCHED} {SAVE} {STOP}
STAGES["train"] = STAGES.get("train", 0) + time.time() - t0
assert os.path.exists(f"{OUT}/{RUN_ID}.json"), "training did not start: read the output above"
print("train", round(STAGES["train"]), "s")'''

EVAL = '''# Score your model and GPT-2 in the same harness: HellaSwag, ARC-Easy, PIQA and LAMBADA. --val first checks that the wrapper gives the native model's loss.
W = OUT + "/weights/weights_step_%06d.pt" % LAST_STEP
if os.path.exists(W):
    t0 = time.time()
    !python eval.py --weights {W} --baseline --val {DATA}/val.bin --out evals
    STAGES["eval"] = time.time() - t0
    assert len(glob.glob("evals/*.json")) == 2, "scoring failed for your model or for GPT-2: read the output above"
    print("eval", round(STAGES["eval"]), "s")
else:
    print("No weights for step", LAST_STEP, "yet: training has not finished. Rerun the training cell first.")'''

TABLE = '''# Results next to GPT-2, in percent. The page's budget table has our reference models to compare with
KEYS = {"hellaswag": "acc_norm,none", "arc_easy": "acc,none", "piqa": "acc,none", "lambada_openai": "acc,none"}
rows = {}
for f in sorted(glob.glob("evals/*.json")):
    ev = json.load(open(f))
    rows[ev["runId"].replace("lltCapstoneGPT2-eval-", "")] = {t: round(100 * ev["evals"][t][k], 1) for t, k in KEYS.items() if t in ev["evals"]}
for name, r in rows.items():
    print(f"{name:<24}", "  ".join(f"{t} {v}" for t, v in r.items()))'''

ACCOUNT = '''# Compute accounting: FLOPs per token (6N + 12*L*d*T), total FLOPs, MFU against the T4's 65 TFLOPS fp16 spec, and GPU-hours
import statistics
run = json.load(open(f"{OUT}/{RUN_ID}.json"))
N, m = run["config"]["params"], run["config"]["model"]
fpt = 6 * N + 12 * m["L"] * m["d"] * m["ctx"]
mine = [x for x in run["metrics"] if x["step"] > START_STEP]
steady = mine[5:] or mine                                      # the first steps include the torch.compile warmup
tokens = len(mine) * run["config"]["tokens_per_step"]
wall0 = run["metrics"][START_STEP - 1]["wallSec"] if START_STEP else 0.0
train_sec = mine[-1]["wallSec"] - wall0                        # includes compile, validation passes, samples and checkpoint saves
acct = {"gpu": run["hardware"]["gpu"], "gpus": run["hardware"]["count"], "stepsTrained": len(mine), "tokensTrained": tokens,
        "flopsPerToken": fpt, "totalFlops": tokens * fpt,
        "medianTokPerSec": statistics.median(x["tokPerSec"] for x in steady),
        "medianMfuVsT4Spec65": statistics.median(x["mfu"] for x in steady),
        "peakMemGBPerGpu": max(x["memGB"] for x in mine), "trainWallSec": round(train_sec, 1),
        "trainGpuHours": round(train_sec * run["hardware"]["count"] / 3600, 3),
        "stageSec": {k: round(v, 1) for k, v in STAGES.items()},
        "stageGpuHours": {k: round(STAGES[k] * run["hardware"]["count"] / 3600, 3) for k in ("download", "train", "eval") if k in STAGES},
        "finalValLoss": next((x["valLoss"] for x in reversed(run["metrics"]) if "valLoss" in x), None)}
json.dump(acct, open("compute_accounting.json", "w"), indent=1)
print(json.dumps(acct, indent=1))
print("Idle time counts too: every second the GPUs sat waiting on prep, downloads or you is GPU-time you paid for or used from your quota.")'''

CELLS = [("md", INTRO), ("md", None), ("code", RUNTIME),
         ("md", "## Install"), ("code", INSTALL),
         ("md", "## Choose the path and get the code\n\nThe settings cell holds the flags of the page's commands. The default is the quick path."),
         ("code", SETTINGS), ("code", GET_CODE),
         ("md", "## Prepare the data\n\nFile 0 gives the fixed validation slice. On the quick path, files 12 and 13 hold text our run reached only after step 17,000, so the decay trains on text the checkpoint has not seen."),
         ("code", PREP),
         ("md", "## Download our step 4,529 checkpoint (quick path)"), ("code", FETCH),
         ("md", "## Train\n\nDDP over the GPUs with fp16 autocast and a GradScaler, micro-batch 8. The log should say `same layout: False` on the quick path, because our checkpoint came from four A100s. That is expected."),
         ("code", TRAIN),
         ("md", "## Score the model and GPT-2 in lm-evaluation-harness"), ("code", EVAL), ("code", TABLE),
         ("md", "## Account for the compute"), ("code", ACCOUNT)]


def cell(kind, src):
    if kind == "md" and src is None:
        return {"cell_type": "markdown", "metadata": {}, "source": HELP}
    lines = src.split("\n")
    body = [x + "\n" for x in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body}


nb = {"cells": [cell(k, s) for k, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}
out = HERE / "lltCapstoneGPT2.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(nb["cells"]))
