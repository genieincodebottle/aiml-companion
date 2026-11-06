# build_notebook.py - the learner notebook lltMemoryThroughput.ipynb, built from the same step files the page shows.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        return f.read().rstrip() + "\n"


def wf(name):
    return "%%writefile " + name + "\n" + src("steps/" + name)


def bang(rel):
    return "".join("!" + line + "\n" for line in src(rel).splitlines() if line.strip() and not line.startswith("#"))


CELLS = [
    ("md", "# Make Training Fit: Memory, Speed and Two GPUs\n\nLLM Training Lab, module 8. Run this on Kaggle with **Settings > Accelerator > GPU T4 x2**. Parts 1 and 3 also run on one Colab T4; part 2 needs two GPUs.\n\nEvery cell matches a step on the lab page. Our recorded outputs are in `runs/lltMemoryThroughput/` of this repository."),
    ("code", src("steps/check_runtime.py")),
    ("md", "## Part 1. Memory and speed on one T4"),
    ("code", wf("gpt.py")),
    ("code", src("steps/s01_static.py")),
    ("code", src("steps/s02_predict.py")),
    ("code", src("steps/s03_grid.py")),
    ("code", src("steps/s04_timing.py")),
    ("code", src("steps/s05_cut.py")),
    ("code", src("steps/s06_precision.py")),
    ("md", "## Part 2. Two GPUs\n\nEach script runs in its own processes under `torchrun`, so the notebook writes it to a file first. Restart the kernel before this part if part 1 ran in the same session, so part 1's tensors do not hold GPU 0's memory."),
    ("code", wf("allreduce.py")),
    ("code", "!torchrun --nproc_per_node=2 allreduce.py\n"),
    ("code", wf("ddp.py")),
    ("code", bang("steps/launch_ddp.sh")),
    ("code", wf("fsdp.py")),
    ("code", "!torchrun --nproc_per_node=1 fsdp.py\n!torchrun --nproc_per_node=2 fsdp.py\n"),
    ("code", wf("fsdp_ckpt.py")),
    ("code", "!torchrun --nproc_per_node=2 fsdp_ckpt.py\n!rm -rf /tmp/ckpt_fsdp /tmp/rank0_only.pt\n"),
    ("md", "## Part 3. When training blows up"),
    ("code", "# The versions part 3 was checked with. prep.py runs as its own process, so no restart is needed.\n!pip install -q \"datasets==5.1.0\" \"tokenizers==0.23.3\"\n"),
    ("code", wf("prep.py")),
    ("code", wf("tiny.py")),
    ("code", wf("sweep.py")),
    ("code", bang("steps/launch_p3.sh")),
    ("code", src("steps/read_sweep.py")),
]

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s} for k, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(HERE, "lltMemoryThroughput.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote lltMemoryThroughput.ipynb,", len(CELLS), "cells")
