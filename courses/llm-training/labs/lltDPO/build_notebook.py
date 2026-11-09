"""Build the learner notebook lltDPO.ipynb and the Kaggle kernels from steps/cellNN_stepK.py.

The page's code blocks come from the same files (export_page_code.py), so page and notebook cannot disagree.
Kernels differ from the learner notebook only in the flags they set and, for the smoke, in the data sizes:
  llt-dpo-smoke  toy sizes, every RUN_* flag True (one code path check, no numbers)
  llt-dpo-core   RUN_SFT, RUN_DPO and step 9 True: trains the SFT model, the main DPO run and the beta sweep
  llt-dpo-exp    learner defaults (downloads our SFT model and DPO adapter), steps 10 and 11 True
  python build_notebook.py
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent.parent / "kaggle"
STEPS = {1: "Step 1. Derive the DPO loss (NumPy)",
         2: "Step 2. Score a reply with a masked sequence log-prob",
         3: "Step 3. Match DPOTrainer's loss on one batch",
         4: "Step 4. Build the SFT starting point (RUN_SFT = False downloads ours)",
         5: "Step 5. Train DPO by hand (RUN_DPO = False downloads our adapter)",
         6: "Step 6. Read the log-probs, not only the loss",
         7: "Step 7. Judge the policy with a gold reward model",
         8: "Step 8. Run the same training with DPOTrainer",
         9: "Step 9. Sweep beta (RUN_EXPERIMENT = True trains three more models)",
         10: "Step 10. Compare IPO, SimPO, ORPO and KTO (RUN_EXPERIMENT = True trains four models)",
         11: "Step 11. Pairs from your own model, labelled by two reward models (RUN_EXPERIMENT = True)"}

INTRO = ("# DPO: derive it, implement it, stress it\n\nLLM Training Lab, module 18. Regime PREFERENCE: a LoRA adapter on an SFT'd "
         "Qwen3-0.6B-Base, trained with the DPO loss on UltraFeedback preference pairs, against a frozen reference.\n\n"
         "Run the cells in order on one T4 (Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2). "
         "Steps 1 to 3 also run on a CPU. Code cell numbers match the lab page.").replace("\n", chr(10))

RUNTIME = (HERE.parent / "lltRewardModels" / "build_notebook.py").read_text(encoding="utf-8").split("RUNTIME = '''")[1].split("'''")[0]
RUNTIME = RUNTIME.replace("steps 1 to 4 still run. For step 5 on", "steps 1 to 3 still run. For step 4 on").replace("< 6", "< 12").replace("at least 6 GB", "at least 12 GB")

INSTALL = '''# Pinned Hugging Face stack of our recorded run. torch and numpy stay as the runtime ships them.
%pip uninstall -y -q torchao
%pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0'''

TEE = '''import os, sys, time
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()   # Kaggle, or any Jupyter host
class _Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "a"), sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x); self.f.flush()
    def flush(self):
        self.o.flush(); self.f.flush()
    def __getattr__(self, name):
        return getattr(self.o, name)
sys.stdout = _Tee(os.path.join(WORK, "notebook_stdout.txt"))
T_START = time.time()
CELL_TIMES = {}'''

RECORD = '''import json, torch, transformers, trl, peft, datasets, platform
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in ("dc", "dr")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    return o
rec = {"kernel": KERNEL, "gpu": torch.cuda.get_device_name(0), "env": {"python": platform.python_version(), "torch": torch.__version__,
       "cuda": torch.version.cuda, "transformers": transformers.__version__, "trl": trl.__version__, "peft": peft.__version__,
       "datasets": datasets.__version__}, "cellTimes": CELL_TIMES, "runs": _clean(runs), "totalSec": round(time.time() - T_START, 1)}
for name in ("gens", "gold", "judge2"):
    if name in globals():
        rec[name] = _clean(globals()[name])
if "final" in globals():
    rec["finalHeld"] = {"dc": final["dc"].tolist(), "dr": final["dr"].tolist()}
json.dump(rec, open(os.path.join(WORK, "llt_dpo_" + KERNEL + ".json"), "w"), indent=1)
print("RECORD written", rec["totalSec"])'''

SMOKE = [("N_SFT, N_DPO, N_HELD, N_EVAL = 3000, 1000, 300, 200", "N_SFT, N_DPO, N_HELD, N_EVAL = 48, 24, 16, 8"),
         ("max_steps=40,", "max_steps=4,"), ("every=25", "every=2"), ("% 75 == 0", "% 2 == 0")]


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


def build(kernel=None, flags=None, replace=()):
    cells = [("md", INTRO), ("code", RUNTIME), ("code", INSTALL)]
    if kernel:
        cells.append(("code", TEE + chr(10) + "KERNEL = " + repr(kernel)))
    seen = set()
    for i, path in enumerate(sorted((HERE / "steps").glob("cell*_step*.py")), start=1):
        step = int(path.stem.split("_step")[1])
        if step not in seen:
            seen.add(step)
            cells.append(("md", "## " + STEPS[step] + chr(10) * 2 + "Code cell " + str(i) + " on the lab page."))
        src = path.read_text(encoding="utf-8").rstrip()
        if kernel:
            for flag, value in (flags or {}).get(path.stem, {}).items():
                old = flag + " = False"
                assert old in src, (path.stem, flag)
                src = src.replace(old, flag + " = " + value, 1)
            for a, b in replace:
                src = src.replace(a, b)
            src = ('_t = time.time(); print("CELL ' + path.stem + ' start", round(_t - T_START, 1))' + chr(10) + src + chr(10) +
                   'CELL_TIMES["' + path.stem + '"] = round(time.time() - _t, 1); print("CELL ' + path.stem + ' end", round(time.time() - T_START, 1))')
        cells.append(("code", src))
    if kernel:
        cells.append(("code", RECORD))
    return cells


def write_kernel(slug, title, cells):
    kdir = KAGGLE / slug
    os.makedirs(kdir, exist_ok=True)
    with open(kdir / (slug + ".ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1)
    meta = {"id": "genieincodebottle/" + slug, "title": title, "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [], "keywords": []}
    with open(kdir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("kernel", kdir)


ALL_TRUE = {"cell05_step4": {"RUN_SFT": "True"}, "cell06_step5": {"RUN_DPO": "True"}, "cell10_step9": {"RUN_EXPERIMENT": "True"},
            "cell11_step10": {"RUN_EXPERIMENT": "True"}, "cell12_step11": {"RUN_EXPERIMENT": "True"}}
CORE = {"cell05_step4": {"RUN_SFT": "True"}, "cell06_step5": {"RUN_DPO": "True"}, "cell10_step9": {"RUN_EXPERIMENT": "True"}}
EXP = {"cell11_step10": {"RUN_EXPERIMENT": "True"}, "cell12_step11": {"RUN_EXPERIMENT": "True"}}

learner = build()
with open(HERE / "lltDPO.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb(learner), f, indent=1)
write_kernel("llt-dpo-smoke", "LLT DPO Smoke", build("smoke", ALL_TRUE, SMOKE))
write_kernel("llt-dpo-core", "LLT DPO Core", build("core", CORE))
write_kernel("llt-dpo-exp", "LLT DPO Exp", build("exp", EXP))
print("learner cells", len(learner))
