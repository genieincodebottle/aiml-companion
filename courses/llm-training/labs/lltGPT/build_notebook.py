"""Build lltGPT.ipynb from the step cells the lab page shows (steps/*.py).

The page's code blocks are assembled from the same files, and tools check the two are identical,
so the page and the notebook cannot drift. The notebook's first cell is the runtime check.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def code(name):
    return (HERE / "steps" / f"{name}.py").read_text(encoding="utf-8").rstrip("\n")


runtime_check = '''# Runtime check. Run this first. It prints the fix for each problem it finds.
import platform, shutil, sys
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("FIX: transformers 5.22.1 needs Python 3.11 or later. On Colab, keep the default runtime.")
try:
    import psutil
    ram = psutil.virtual_memory().total / 1e9
    print(f"RAM {ram:.1f} GB")
    if ram < 4:
        print("FIX: this lab peaked at PEAK GB of RAM in our run. Use a machine or runtime with 4 GB or more.")
except ImportError:
    print("psutil not installed, RAM not checked")
free = shutil.disk_usage(".").free / 1e9
print(f"Free disk {free:.1f} GB")
if free < 2:
    print("FIX: the GPT-2 download needs 0.55 GB plus cache space. Free some disk space.")
try:
    import torch
    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
except ImportError:
    print("FIX: install PyTorch, pip install torch")
try:
    import numpy
    print("numpy", numpy.__version__, "found | our CPU runs used 2.5.4")
except ImportError:
    print("FIX: install NumPy, pip install numpy")
print("This lab runs on a CPU. No GPU is needed.")'''

peak_ram = '''# Peak RAM of this notebook so far
import psutil, sys
mi = psutil.Process().memory_info()
peak = getattr(mi, "peak_wset", None)  # Windows
if peak is None:
    import resource  # Linux reports ru_maxrss in KB, macOS in bytes
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = r * 1024 if sys.platform.startswith("linux") else r
print(f"Peak RAM {peak / 1e9:.2f} GB")'''


def build(peak_gb):
    md, cc = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# lltGPT, Assemble a Modern GPT\n\nLLM Training Lab, module 4. This notebook runs every step of the lab page "
           "on a CPU. No GPU is needed. Our recorded runs are in `courses/llm-training/runs/lltGPT/`."),
        cc(runtime_check.replace("PEAK", str(peak_gb))),
        md("Install the pinned versions. PyTorch and NumPy are already installed on Colab and Kaggle, so keep the image's versions. Our CPU figures were measured on torch 2.15.0 and numpy 2.5.4."),
        cc('%pip install -q "transformers==5.22.1" "tokenizers==0.23.3" "huggingface_hub==2.4.0"'),
        md("## Step 1. Check the RoPE property\n\nThe same NumPy code the page runs in your browser."),
        cc(code("step1")),
        md("## Step 2. Write the config and the two norms"),
        cc(code("step2")),
        md("## Step 3. Write attention, the two MLPs and the block"),
        cc(code("step3")),
        md("## Step 4. Assemble the GPT, tie the head and scale the init"),
        cc(code("step4")),
        md("## Step 5. Test shapes and causal leakage"),
        cc(code("step5")),
        md("## Step 6. Count the parameters and the FLOPs per token"),
        cc(code("step6")),
        cc(code("step6b")),
        md("## Step 7. Load GPT-2 124M and match Hugging Face's logits"),
        cc(code("step7")),
        md("## Step 8. Toggle each modern component\n\nPredict first. Which swap back to GPT-2's choice raises the "
           "validation loss most after 600 steps? Each run takes several minutes on a laptop CPU."),
        cc(code("step8")),
        cc(code("step8b")),
        cc(peak_ram),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = HERE / "lltGPT.ipynb"
    nbf.write(nb, str(out))
    print("wrote", out, len(nb.cells), "cells")


if __name__ == "__main__":
    import json
    runs = HERE.parent.parent / "runs" / "lltGPT" / "lltGPT-t1-steps-v2026-11.json"
    build(json.loads(runs.read_text(encoding="utf-8"))["evals"]["peakRamGB"])
