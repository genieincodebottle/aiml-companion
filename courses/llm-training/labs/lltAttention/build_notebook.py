"""Build lltAttention.ipynb from the page cells in cells/*.py (the same files the lab page embeds).

Usage: python build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def code(name):
    return (HERE / "cells" / (name + ".py")).read_text(encoding="utf-8").rstrip("\n")


RUNTIME_CHECK = '''# Runtime check. Run this first. It prints what to fix before the lab cells run.
import platform, shutil, sys
ok = True
if sys.version_info < (3, 11):
    ok = False
    print("Python", platform.python_version(), "is too old. Use 3.11 or later (Colab: Runtime > Change runtime type).")
try:
    import torch, numpy
    print("python", platform.python_version(), "| torch", torch.__version__, "| numpy", numpy.__version__)
except ImportError as e:
    ok = False
    print("Missing package:", e.name, "| install it with: pip install torch==2.15.0 numpy")
try:
    import psutil
    ram = psutil.virtual_memory().total / 1e9
    print(f"RAM {ram:.1f} GB")
    if ram < 6:
        print("Step 6 needs about 3 GB free at T = 4,096. Close other programs or lower the largest T.")
except ImportError:
    pass
free = shutil.disk_usage(".").free / 1e9
print(f"free disk {free:.1f} GB")
if free < 1:
    ok = False
    print("Free at least 1 GB of disk for the 22 MB TinyStories download in step 8.")
gpu = "torch" in globals() and torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if gpu else "none. Steps 1 to 6 and 8 run on the CPU. Step 7 needs a GPU "
      "(Colab: Runtime > Change runtime type > T4 GPU).")
print("Runtime OK" if ok else "Fix the lines above, then run this cell again.")'''

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell("# lltAttention, Causal Self-Attention, Built and Tested\n\nLLM Training Lab, module 3. "
                             "The cells below are the lab page's code, in the same order. Steps 1 to 3 need only NumPy. "
                             "Steps 4 to 6 and 8 run on a CPU. Step 7 needs a CUDA GPU, such as a free Colab T4.\n\n"
                             "Recorded results for every step: `../../runs/lltAttention/`."),
    nbf.v4.new_code_cell(RUNTIME_CHECK),
    nbf.v4.new_markdown_cell("## Step 1. Write single-head causal attention in NumPy"),
    nbf.v4.new_code_cell(code("step1")),
    nbf.v4.new_markdown_cell("## Step 2. Split the width into heads"),
    nbf.v4.new_code_cell(code("step2")),
    nbf.v4.new_markdown_cell("## Step 3. Check the 1/sqrt(d) scale with numbers"),
    nbf.v4.new_code_cell(code("step3")),
    nbf.v4.new_markdown_cell("## Step 4. Build the PyTorch module and match SDPA"),
    nbf.v4.new_code_cell(code("step4")),
    nbf.v4.new_markdown_cell("## Step 5. Prove the layer cannot see ahead"),
    nbf.v4.new_code_cell(code("step5")),
    nbf.v4.new_markdown_cell("## Step 6. Measure attention memory on the CPU"),
    nbf.v4.new_code_cell(code("step6")),
    nbf.v4.new_markdown_cell("## Step 7. Measure naive attention and SDPA on a T4\n\nSkip this cell on a CPU runtime. "
                             "The table on the lab page is our run on a Kaggle T4."),
    nbf.v4.new_code_cell(code("step_t4")),
    nbf.v4.new_markdown_cell("## Step 8. Train a small model with a broken mask\n\nUses the classes from steps 4 and 5."),
    nbf.v4.new_code_cell(code("step7")),
    nbf.v4.new_markdown_cell("## Prove it\n\nFill in the four functions, then run the cell. It prints `All checks passed` when they are right."),
    nbf.v4.new_code_cell(code("challenge_starter")),
]
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = HERE / "lltAttention.ipynb"
nbf.write(nb, str(out))
print("wrote", out, len(cells), "cells")
