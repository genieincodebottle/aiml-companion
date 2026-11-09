"""Build the learner notebook lltDistillation.ipynb and the Kaggle kernels from steps/cellNN_stepK.py.

The page's code blocks come from the same files (export_page_code.py), so the page and the notebook cannot disagree.
Kernel variants change only the flag and size lines listed in VARIANTS, by exact text replacement:
  python build_notebook.py              -> lltDistillation.ipynb (learner) and all kernel folders in ../../kaggle/
  smoke    llt-distill-smoke     tiny sizes, every branch except USE_OUR_TEACHER_DATA / USE_OUR_ADAPTER = True
  main     llt-distill-main      the Run Library run: the teacher writes the data, every arm of the grid, two seeds
  default  llt-distill-notebook  the learner defaults, plus USE_OUR_ADAPTER = True (downloads ours from the Hub)
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "Step 1. Fit one Gaussian to two modes, both ways (NumPy, CPU)",
         2: "Step 2. Write the divergence and match TRL's kernel (CPU)",
         3: "Step 3. Load the teacher and the student, and measure both on GSM8K",
         4: "Step 4. Let the teacher write the training answers",
         5: "Step 5. Off-policy, SFT on the teacher's answers",
         6: "Step 6. Logit KD, forward KL over the whole vocabulary",
         7: "Step 7. On-policy, the student samples and the teacher grades every token",
         8: "Step 8. Separate the source of the answers from the KL direction (RUN_EXPERIMENT)",
         9: "Step 9. Compare the methods with paired intervals and two seeds"}

INTRO = ("# Distillation for LLMs, off-policy, on-policy and logits" + chr(10) * 2 +
         "LLM Training Lab, module 20. Regime DISTILLATION: teacher Qwen3-4B-Base in 4-bit NF4 (frozen), student Qwen3-0.6B-Base "
         "with LoRA adapters, trained on GSM8K questions three ways." + chr(10) * 2 +
         "Run the cells in order on one T4 (Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2). "
         "Steps 1 and 2 also run on a CPU. Code cell numbers match the lab page.")

RUNTIME = '''import os, sys, platform, shutil, subprocess
from importlib.metadata import version, PackageNotFoundError
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # one GPU; set before torch loads
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
    problems.append("No GPU: steps 1 and 2 still run. For step 3 on, Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    problems.append("A P100 has no kernels in current PyTorch. Pick a T4.")
cache = os.path.expanduser("~/.cache")
os.makedirs(cache, exist_ok=True)
free_disk = shutil.disk_usage(cache).free / 1e9
if free_disk < 15:
    problems.append("Free at least 15 GB of disk for the two models and GSM8K (" + str(round(free_disk)) + " GB free).")
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
%pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 bitsandbytes==0.50.2 tokenizers==0.23.3 huggingface_hub==2.4.0'''

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
sys.stdout = _Tee("notebook_stdout.txt")
T_START = time.time()'''

PEAK = '''import torch, json
print("KERNEL peak GB over the whole session", round(torch.cuda.max_memory_allocated() / 1e9, 2))'''

VARIANTS = {
    "smoke": {"slug": "llt-distill-smoke", "title": "LLT Distill Smoke", "replace": [
        ("N_TEST, N_TRAIN = 500, 2048", "N_TEST, N_TRAIN = 32, 64"),
        ("STEPS, BATCH, MICRO, LR = 64, 32, 4, 2e-4", "STEPS, BATCH, MICRO, LR = 2, 16, 4, 2e-4"),
        ("USE_OUR_TEACHER_DATA = True", "USE_OUR_TEACHER_DATA = False"),
        ("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")]},
    "main": {"slug": "llt-distill-main", "title": "LLT Distill Main", "replace": [
        ("USE_OUR_TEACHER_DATA = True", "USE_OUR_TEACHER_DATA = False"),
        ("RUN_EXPERIMENT = False", "RUN_EXPERIMENT = True")]},
    "default": {"slug": "llt-distill-notebook", "title": "LLT Distill Notebook", "replace": [
        ("USE_OUR_ADAPTER = False", "USE_OUR_ADAPTER = True")]},
}


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


def build(variant=None):
    cells = [("md", INTRO), ("code", RUNTIME), ("code", INSTALL)]
    if variant:
        cells.append(("code", TEE))
    seen = set()
    expect = json.load(open(HERE / "expect.json", encoding="utf-8"))     # the page's "What you should see", per step
    paths = sorted((HERE / "steps").glob("cell*_step*.py"))
    for i, path in enumerate(paths, start=1):
        step = int(path.stem.split("_step")[1])
        if step not in seen:
            seen.add(step)
            cells.append(("md", "## " + STEPS[step] + chr(10) * 2 + "Code cell " + str(i) + " on the lab page."))
        if i > 1 and int(paths[i - 2].stem.split("_step")[1]) != step:
            cells.insert(len(cells) - 1, ("md", "**What you should see.** " + expect[str(step - 1)]))
        src = path.read_text(encoding="utf-8").rstrip()
        if variant:
            for old, new in VARIANTS[variant]["replace"]:
                src = src.replace(old, new)
            src = 'print("CELL ' + path.stem + ' start", round(time.time() - T_START, 1))' + chr(10) + src + chr(10) + \
                  'print("CELL ' + path.stem + ' end", round(time.time() - T_START, 1))'
        cells.append(("code", src))
    cells.append(("md", "**What you should see.** " + expect[str(step)]))
    if variant:
        cells.append(("code", PEAK))
    return cells


if __name__ == "__main__":
    learner = build()
    with open(HERE / "lltDistillation.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb(learner), f, indent=1)
    for variant, v in VARIANTS.items():
        cells = build(variant)
        joined = chr(10).join(src for kind, src in cells if kind == "code")
        for old, new in v["replace"]:
            assert new in joined, (variant, "replacement did not apply", old)
        kdir = HERE.parent.parent / "kaggle" / v["slug"]
        os.makedirs(kdir, exist_ok=True)
        with open(kdir / (v["slug"] + ".ipynb"), "w", encoding="utf-8") as f:
            json.dump(nb(cells), f, indent=1)
        meta = {"id": "genieincodebottle/" + v["slug"], "title": v["title"], "code_file": v["slug"] + ".ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": [], "keywords": []}
        with open(kdir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print(variant, kdir)
    print("learner cells", len(learner))