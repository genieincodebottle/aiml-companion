"""Build lltCapstoneReasoner.ipynb, the T2 notebook (one T4 on Kaggle or Colab). Run: python build_notebook.py"""
import json
from pathlib import Path

CELLS = [
    ("md", "# Capstone: train a small reasoner with RL (T2 path)\n\nOne T4, Qwen3-0.6B-Base, GRPO on Countdown. Run the cells from the top. "
           "Each session trains one cell of the ablation until `MAX_MINUTES`, saves, and stops. Run the notebook again to continue.\n\n"
           "Kaggle: Settings > Accelerator > GPU T4, and Internet on. Colab: Runtime > Change runtime type > T4 GPU."),
    ("code", "import os, sys, shutil, subprocess\n"
             "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
             "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
             "assert sys.version_info >= (3, 11), 'Python 3.11 or newer is needed'\n"
             "try:\n    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout)\n"
             "except FileNotFoundError:\n    print('No GPU found. Runtime > Change runtime type > T4 GPU (Colab), or Accelerator > GPU T4 (Kaggle)')\n"
             "print('free disk GB', round(shutil.disk_usage('.').free / 1e9, 1))"),
    ("code", "!pip uninstall -y -q torchao\n"
             "!pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 math-verify==0.9.0"),
    ("code", "WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/content'\n"
             "os.chdir(WORK)\n"
             "if not os.path.isdir('aiml-companion'):\n"
             "    !git clone -q --depth 1 --filter=blob:none --sparse https://github.com/genieincodebottle/aiml-companion.git\n"
             "    !cd aiml-companion && git sparse-checkout set courses/llm-training/labs/lltCapstoneReasoner courses/llm-training/labs/lltRL\n"
             "LAB = os.path.join(WORK, 'aiml-companion/courses/llm-training/labs/lltCapstoneReasoner')\n"
             "os.chdir(LAB)\n"
             "!python test_capstone.py"),
    ("md", "## Choose the cell this session trains\n\nThe ablation has 4 cells x 3 seeds (our reference ran 3 seeds per cell). Keep `OUT` on storage that survives the session "
           "(Kaggle: save the output as a dataset or download `runs/`; Colab: mount Google Drive and point `RUNS` there)."),
    ("code", "AGG, SCALE, SEED = 'grpo', 1, 0          # agg: grpo | dr_grpo ; scale: 1 | 0 ; seed: 0 | 1\n"
             "MAX_MINUTES = 100                         # stop cleanly before the session limit, then resume\n"
             "RUNS = os.path.join(WORK, 'runs')\n"
             "NAME = 'agg-%s_scale-%d_s%d' % (AGG, SCALE, SEED)\n"
             "OUT = os.path.join(RUNS, NAME)\n"
             "print(OUT)"),
    ("code", "# Base model first (once): pass@1 to pass@16 on the 200 held-out problems\n"
             "if not os.path.exists(os.path.join(WORK, 'evals/base.json')):\n"
             "    !python evaluate.py --out {WORK}/evals/base.json --n 16 --batch 64"),
    ("code", "!python train.py --out {OUT} --agg {AGG} --scale {SCALE} --seed {SEED} --max_minutes {MAX_MINUTES}"),
    ("code", "# After the run has finished (done.json exists), evaluate it\n"
             "if os.path.exists(os.path.join(OUT, 'done.json')):\n"
             "    !python evaluate.py --out {WORK}/evals/{NAME}.json --adapter {OUT}/adapter-final --n 16 --batch 64\n"
             "else:\n"
             "    print('Not finished yet. Run the notebook again to continue training.')"),
    ("code", "# When all 8 evaluations exist\n"
             "!python analyze.py --runs {RUNS} --evals {WORK}/evals --out {WORK}/results.json"),
]


def cell(kind, src):
    lines = src.split("\n")
    body = [x + "\n" for x in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "lltCapstoneReasoner.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
