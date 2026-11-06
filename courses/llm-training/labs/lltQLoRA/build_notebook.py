"""Build the learner notebook lltQLoRA.ipynb (md + code cells) and the Kaggle measurement kernel
(the same cells plus the recorder cells) from cells_lab.py.

  python build_notebook.py            -> lltQLoRA.ipynb and ../../kaggle/llt-qlora-lab/
"""
import json, os, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = runpy.run_path(os.path.join(HERE, "cells_lab.py"))["CELLS"]


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


STEPS = {1: "Step 1. Check the runtime and pin the libraries", 4: "Step 2. Quantize one real Qwen3.5 weight matrix to NF4 by hand",
         6: "Step 3. Match bitsandbytes and count the bits per weight", 7: "Step 4. Load Qwen3.5-9B-Base in 4 bits",
         8: "Step 5. Add LoRA adapters without the fp32 upcast", 9: "Step 6. Measure memory and speed at 512, 1,024 and 2,048 tokens",
         10: "Step 7. Build the intent task and score the base model", 13: "Step 8. Fine-tune with SFTTrainer and prove the loss",
         18: "Step 9. Resume after the session ends", 19: "Step 10. Tune on a budget with a 0.8B proxy (about 62 min; set USE_OUR_STUDY = True to skip)",
         22: "Step 11. Compare NF4, a 16-bit base and DoRA on the proxy (about 77 min; one seed halves it)",
         23: "Step 12. Apply the tuned settings to the 9B (about 56 min; optional)"}
learner, n = [], 0
for k, s in CELLS:
    if k == "code":
        n += 1
        if n in STEPS:
            learner.append(("md", "## " + STEPS[n] + chr(10) * 2 + "Code cell " + str(n) + " onward, as numbered on the lab page."))
        learner.append((k, s))
    elif k == "md":
        learner.append((k, s))
with open(os.path.join(HERE, "lltQLoRA.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb(learner), f, indent=1)

kernel = [("code" if k == "rec" else k, s) for k, s in CELLS]
slug = "llt-qlora-lab"
kdir = os.path.join(HERE, "..", "..", "kaggle", slug)
os.makedirs(kdir, exist_ok=True)
with open(os.path.join(kdir, slug + ".ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb(kernel), f, indent=1)
meta = {"id": "genieincodebottle/" + slug, "title": "LLT QLoRA Lab", "code_file": slug + ".ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [], "keywords": []}
with open(os.path.join(kdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=1)
print("learner cells", len(learner), "| kernel cells", len(kernel), "|", kdir)
