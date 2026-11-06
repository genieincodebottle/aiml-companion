# build_notebook.py - the learner notebook lltLoRA.ipynb, built from the same step files the page shows.
# Usage: python build_notebook.py   (writes lltLoRA.ipynb next to this file)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read().rstrip() + "\n"


def wf(name):
    return "%%writefile " + name + "\n" + src("steps/" + name)


INSTALL = ("%pip uninstall -y -q torchao\n"
           "%pip install -q transformers==5.22.1 tokenizers==0.23.3 huggingface_hub==2.4.0 peft==0.21.2 "
           "accelerate==1.15.0 bitsandbytes==0.50.2 datasets==5.1.0\n")

CONSTANTS = open(os.path.join(HERE, "steps", "constants.py"), encoding="utf-8").read().rstrip() + "\n"

CELLS = [
    ("md", "# LoRA From First Principles, Measured Against Full Fine-Tuning\n\nLLM Training Lab, module 12. "
           "Run it on a free T4: Colab (Runtime > Change runtime type > T4 GPU) or Kaggle (Settings > Accelerator > GPU T4 x2). "
           "Every cell matches a step on the lab page. Our recorded outputs are in `runs/lltLoRA/` of this repository.\n\n"
           "Steps 1 to 4 also run on a CPU. Steps 5 to 10 need the GPU."),
    ("code", src("steps/check_runtime.py")),
    ("md", "Install the pinned Hugging Face libraries. Keep the PyTorch and NumPy your runtime already has. "
           "Kaggle's image ships torchao 0.10, which PEFT 0.21.2 refuses, so the cell removes it."),
    ("code", INSTALL),
    ("code", wf("gpt_model.py")),
    ("code", wf("lora_lab.py")),
    ("md", "## Step 1. Write a LoRA linear layer"),
    ("code", src("steps/s01_lora_linear.py")),
    ("md", "## Step 2. Merge and unmerge"),
    ("code", src("steps/s02_merge.py")),
    ("md", "## Step 3. Add LoRA to your GPT"),
    ("code", src("steps/s03_inject_gpt.py")),
    ("md", "## Step 4. Check your layer against PEFT on Qwen3.5-0.8B-Base"),
    ("code", src("steps/s04_peft_check.py")),
    ("md", "## Step 5. Full fine-tuning baseline"),
    ("code", CONSTANTS),
    ("code", src("steps/s05_full_ft.py")),
    ("md", "## Step 6. Sweep the full fine-tune's learning rate\n\nChange `FULL_LR` in the settings cell and run step 5 again, once per rate. Our runs used 3e-6, 1e-5, 3e-5 and 1e-4."),
    ("md", "## Step 7. The SVD of the full fine-tune's update"),
    ("code", src("steps/s06_svd.py")),
    ("md", "## Step 8. LoRA on all linear layers"),
    ("code", src("steps/s07_lora.py")),
    ("md", "## Step 9. Sweep the LoRA learning rate\n\nCall `lora_run(lr=...)` once per rate, for example `m = lora_run(lr=1e-4)`, then `del m` and `torch.cuda.empty_cache()`. Our runs used 3e-5, 1e-4, 3e-4, 1e-3 and 3e-3."),
    ("md", "## Step 10. Rank and target layers"),
    ("code", src("steps/s08_sweep.py")),
]

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s} for k, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(HERE, "lltLoRA.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote lltLoRA.ipynb,", len(CELLS), "cells")
