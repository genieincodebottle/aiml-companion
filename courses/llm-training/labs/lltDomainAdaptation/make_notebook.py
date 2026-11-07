"""Build the learner notebook lltDomainAdaptation.ipynb (one T4, Colab or Kaggle) and its Kaggle test kernel.
Usage: python make_notebook.py <quick_steps>
The notebook writes dalab.py, extend_tokenizer.py and qa_split.json into the runtime, so it needs nothing else.
Its code cells are page_blocks.py, the same text the page shows."""
import json, os, subprocess, sys
import page_blocks as B

HERE = os.path.dirname(os.path.abspath(__file__))
KAG = os.path.join(HERE, "..", "..", "kaggle")
QUICK = sys.argv[1] if len(sys.argv) > 1 else "150"
src = lambda p: open(os.path.join(HERE, p), encoding="utf-8").read()

C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s.replace("QUICK_STEPS_VALUE", QUICK)))

md("# Adapt a model to Hindi with continued pretraining, fine-tuning or RAG\n\n"
   "LLM Training Lab, lltDomainAdaptation. One free T4 GPU (Colab or Kaggle). Run the cells in order.\n\n"
   "Base model: Qwen3-0.6B-Base. Qwen3.5-0.8B-Base is newer; on a free T4 its full-parameter step ran 4.55 s at a "
   "15.17 GB peak vs 3.81 s at 11.91 GB for Qwen3-0.6B-Base in run llt-model-probe (2026-11-06), so the continued "
   "pretraining here uses Qwen3-0.6B-Base.")
code('''# Runtime check. Run this cell first. It imports nothing that the next cell installs.
import os, platform, shutil, subprocess, sys
from importlib import metadata
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # Kaggle gives two T4s; this lab uses one
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch starts: stops cached GPU blocks fragmenting
print("Python", platform.python_version())
if sys.version_info < (3, 11):
    print("PROBLEM Python 3.11 or later is needed, 3.12 recommended. Start a current Colab or Kaggle session.")
for pkg in ("torch", "numpy", "transformers"):
    try:
        print(pkg, metadata.version(pkg))
    except metadata.PackageNotFoundError:
        print(pkg, "not installed")
free = shutil.disk_usage(".").free / 1e9
print("Free disk", round(free, 1), "GB")
if free < 10:
    print("PROBLEM this lab downloads about 3 GB of models and data and writes a 1.2 GB checkpoint. Free some disk.")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except FileNotFoundError:                          # CPU runtimes have no nvidia-smi
    smi = ""
gpu = smi.splitlines()[0] if smi else ""
print("GPU", gpu or "none")
if not gpu:
    print("PROBLEM steps 1, 3, 4, 6, 7 and 8 need a GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2.")
elif "P100" in gpu:
    print("PROBLEM a P100 has no kernels in current PyTorch. Pick a T4 instead.")
else:
    cap = float(gpu.split(",")[-1])
    print("fp16 autocast with a GradScaler (no bf16 tensor cores below compute capability 8.0)" if cap < 8.0 else "bf16-capable GPU; the cells still use fp16 autocast")
print("Runtime OK" if sys.version_info >= (3, 11) and gpu and "P100" not in gpu else "Fix the PROBLEM lines, then run this cell again.")''')
code('''# Pinned Hugging Face versions of our recorded runs. torch and numpy stay as the runtime ships them.
# Kaggle's image has torchao 0.10, which PEFT 0.21.2 refuses, so remove it first.
%pip uninstall -y -q torchao
%pip install -q "transformers==5.22.1" "peft==0.21.2" "bitsandbytes==0.50.2" "accelerate==1.15.0" "datasets==5.1.0" "tokenizers==0.23.3" "huggingface_hub==2.4.0" "sentence-transformers==6.2.0"''')
code("# The lab library (data, evaluation panel, training loop, interpolation, tokenizer extension) and the question split.\n"
     "FILES = " + repr({"dalab.py": src("dalab.py"), "extend_tokenizer.py": src("extend_tokenizer.py"), "qa_split.json": src("qa_split.json")}) + "\n"
     "for n, t in FILES.items():\n    open(n, 'w', encoding='utf-8').write(t)\nprint('wrote', list(FILES))")
md("## Step 1. Measure the base model on a fixed panel")
code(B.PANEL)
md("## Step 2. Count what Hindi costs in tokens")
code(B.FERT)
md("## Step 3. Baselines: closed book, few-shot and RAG")
code(B.RAG)
code(B.QA)
md("## Step 4. Continued pretraining with re-warm, re-decay and 5% replay\n\n"
   "This cell runs QUICK steps on one T4. Our recorded runs use 320 steps (2.6M tokens); this cell sees about 1.2M. Step 5 of the page compares replay 0, 5 and 25% "
   "and two peak learning rates. Change REPLAY and PEAK_LR to repeat one of them.".replace("QUICK", QUICK))
code(B.CPT)
md("Optional exit. Our 320-step run (`cpt-r5-lr3e-5`) is on the Hugging Face Hub in `aimlcompanion/llm-training-lab`, "
   "folder `lltDomainAdaptation/cpt-r5-lr3e-5`. Set USE_OUR_CHECKPOINT = True to score it without training for 42 minutes. "
   "The cell keeps your own model if the checkpoint is not reachable.")
code('''REPO, SUB = "aimlcompanion/llm-training-lab", "lltDomainAdaptation/cpt-r5-lr3e-5"   # our 320-step checkpoint, fp16, 1.2 GB
USE_OUR_CHECKPOINT = False
if USE_OUR_CHECKPOINT:
    try:
        ours = AutoModelForCausalLM.from_pretrained(REPO, subfolder=SUB, dtype=torch.float32, device_map="cuda:0").eval()
        print("our checkpoint", L.panel(ours, tok, E))
        del ours; torch.cuda.empty_cache()
    except Exception as e:
        print("Checkpoint not reachable:", type(e).__name__, "- keep your own model from the cell above.")
else:
    print("Skipped. Set USE_OUR_CHECKPOINT = True to score our 320-step checkpoint.")''')
md("## Step 6. Interpolate between the base and the continued model")
code(B.INTERP)
md("## Step 7. Extend the tokenizer for Hindi")
code(B.EXT)
code(B.INIT)
md("## Step 8. QLoRA fine-tune on the question-answering format")
code(B.QLORA)
code(B.QLORA_EVAL)


def build(out_nb):
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s} for k, s in C],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out_nb, "w", encoding="utf-8"), indent=1)


build(os.path.join(HERE, "lltDomainAdaptation.ipynb"))
cells_py = os.path.join(KAG, "cells_da_notebook.py")
open(cells_py, "w", encoding="utf-8").write("CELLS = " + repr(C) + "\n")
subprocess.run([sys.executable, os.path.join(KAG, "build_nb.py"), cells_py, os.path.join(KAG, "llt-da-notebook"),
                "llt-da-notebook", "LLT DA Notebook"], check=True)
print("built", len(C), "cells, quick steps", QUICK)
