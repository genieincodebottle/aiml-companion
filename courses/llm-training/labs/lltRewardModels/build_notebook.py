"""Build the learner notebook lltRewardModels.ipynb and the Kaggle kernel llt-rm-notebook from steps/cellNN_stepK.py.

The page's code blocks come from the same files (export_page_code.py), so the page and the notebook cannot disagree.
  python build_notebook.py   -> lltRewardModels.ipynb and ../../kaggle/llt-rm-notebook/
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "Step 1. Turn a preference into a probability (NumPy)",
         2: "Step 2. Put a scalar head on SmolLM2-360M",
         3: "Step 3. Load the preference pairs and look at what they reward",
         4: "Step 4. Match RewardTrainer's loss on the same batch",
         5: "Step 5. Train the reward model on a T4 (15 to 17 min of training; RUN_TRAINING = False loads ours)",
         6: "Step 6. Read held-out accuracy honestly",
         7: "Step 7. Test the length bias (RUN_EXPERIMENT = True trains two more models, about 35 min)",
         8: "Step 8. Best-of-n against a gold reward model",
         9: "Step 9. Score a RewardBench 2 subset"}

INTRO = ("# Preference data and reward models\n\nLLM Training Lab, module 17. Regime PREFERENCE: SmolLM2-360M plus a scalar head, "
         "every weight trained with the Bradley-Terry loss on UltraFeedback preference pairs.\n\nRun the cells in order on one T4 "
         "(Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2). Steps 1 to 4 also run on a CPU. "
         "Code cell numbers match the lab page.").replace("\n", chr(10))

RUNTIME = '''import os, sys, platform, shutil, subprocess
from importlib.metadata import version, PackageNotFoundError
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # one GPU, set before torch loads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
problems = []
if sys.version_info < (3, 11):
    problems.append("Python 3.11 or newer is needed (3.12 recommended). Start a current Colab or Kaggle session.")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:                                   # a CPU runtime has no nvidia-smi
    smi = []
gpu = smi[0] if smi else ""
if not gpu:
    problems.append("No GPU: steps 1 to 4 still run. For step 5 on, Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    problems.append("A P100 has no kernels in current PyTorch. Pick a T4.")
cache = os.path.expanduser("~/.cache")
os.makedirs(cache, exist_ok=True)
free_disk = shutil.disk_usage(cache).free / 1e9
if free_disk < 6:
    problems.append("Free at least 6 GB of disk for the model and data downloads (" + str(round(free_disk)) + " GB free).")
try:
    torch_version = version("torch")
except PackageNotFoundError:
    torch_version = "not installed"
print("Python", platform.python_version(), "| GPU", gpu or "none", "| torch", torch_version, "| disk free", round(free_disk), "GB")
for p in problems:
    print("PROBLEM", p)
print("Runtime OK" if not problems else "Read the PROBLEM lines first.")'''

INSTALL = '''# Pinned Hugging Face stack of our recorded run. torch and numpy stay as the runtime ships them.
%pip uninstall -y -q torchao
%pip install -q transformers==5.22.1 trl==1.17.1 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0'''

TEE = '''import sys, time
class _Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "a"), sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()
    def __getattr__(self, name):          # isatty, fileno, encoding... come from the real stream
        return getattr(self.o, name)
sys.stdout = _Tee("/kaggle/working/notebook_stdout.txt")
T_START = time.time()'''


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


def build(kernel=False):
    cells = [("md", INTRO), ("code", RUNTIME), ("code", INSTALL)]
    if kernel:
        cells.append(("code", TEE))
    seen = set()
    for i, path in enumerate(sorted((HERE / "steps").glob("cell*_step*.py")), start=1):
        step = int(path.stem.split("_step")[1])
        if step not in seen:
            seen.add(step)
            cells.append(("md", "## " + STEPS[step] + chr(10) * 2 + "Code cell " + str(i) + " on the lab page."))
        src = path.read_text(encoding="utf-8").rstrip()
        if kernel:
            src = 'print("CELL ' + path.stem + ' start", round(time.time() - T_START, 1))' + chr(10) + src + chr(10) + \
                  'print("CELL ' + path.stem + ' end", round(time.time() - T_START, 1))'
        cells.append(("code", src))
    return cells


learner = build()
with open(HERE / "lltRewardModels.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb(learner), f, indent=1)
slug = "llt-rm-notebook"
kdir = HERE.parent.parent / "kaggle" / slug
os.makedirs(kdir, exist_ok=True)
with open(kdir / (slug + ".ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb(build(kernel=True)), f, indent=1)
meta = {"id": "genieincodebottle/" + slug, "title": "LLT RM Notebook", "code_file": slug + ".ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [], "keywords": []}
with open(kdir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print("learner cells", len(learner), "|", kdir)
