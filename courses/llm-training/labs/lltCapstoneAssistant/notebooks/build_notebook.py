"""Build lltCapstoneAssistant_T2.ipynb: the free-T4 (QLoRA) path, one cell per pipeline stage.

  python notebooks/build_notebook.py
Every stage is the same script the T3 reference run used, called with --tier T2, so the notebook and the runbook
cannot drift apart. Work goes to WORK, which must survive between sessions (Google Drive on Colab; on Kaggle,
Save Version keeps /kaggle/working, and you attach that output as input to the next session).
"""
import json
import os

MD, CODE = "markdown", "code"
cells = [
    (MD, "# Capstone: Build an Instruction-Following Assistant (T2, one free T4)\n\n"
         "Qwen3-1.7B-Base -> data card -> SFT (QLoRA) -> on-policy pairs -> DPO (QLoRA) -> IFEval, judge win rate, XSTest -> model card.\n\n"
         "Run the cells from the top. Each stage skips work that is already done, so after a timeout reconnect, re-run "
         "the setup cells, and run the stage again: it resumes from its newest checkpoint. The page's runbook gives "
         "the time each stage takes."),
    (CODE, "import os, sys, shutil, subprocess\n"
           "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
           "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
           "assert sys.version_info >= (3, 11), 'Python 3.11 or newer is needed: pick a newer runtime image'\n"
           "try:\n"
           "    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout)\n"
           "except FileNotFoundError:\n"
           "    raise SystemExit('No GPU. Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU T4 x2, Internet on.')\n"
           "print('free disk GB', shutil.disk_usage('/').free // 2**30)"),
    (CODE, "# Pinned versions (checked 2026-11-10). Kaggle's torchao conflicts with PEFT, so it goes first.\n"
           "!pip uninstall -y -q torchao\n"
           "!pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 bitsandbytes==0.50.2 "
           "lm-eval==0.4.13 langdetect immutabledict nltk\n"
           "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]"),
    (CODE, "# The code: the same package the reference run used\n"
           "!git clone -q --depth 1 --filter=blob:none --sparse https://github.com/genieincodebottle/aiml-companion.git\n"
           "!cd aiml-companion && git sparse-checkout set courses/llm-training/labs/lltCapstoneAssistant\n"
           "%cd aiml-companion/courses/llm-training/labs/lltCapstoneAssistant\n"
           "!python tests/test_cpu.py"),
    (CODE, "# Where the work lives. Colab: mount Drive so a new session finds it.\n"
           "if 'google.colab' in sys.modules:\n"
           "    from google.colab import drive; drive.mount('/content/drive')\n"
           "    WORK = '/content/drive/MyDrive/lltCapstoneAssistant/work'\n"
           "else:\n"
           "    WORK = '/kaggle/working/work'      # attach the previous version's output and copy it here to resume\n"
           "os.makedirs(WORK, exist_ok=True)\n"
           "A = f'--tier T2 --work {WORK}'"),
    (MD, "## Stage 1. Data and the data card (CPU, downloads about 3 GB)"),
    (CODE, "!python -m capstone.data $A\n!cat {WORK}/data/card.json"),
    (MD, "## Stage 2. SFT with QLoRA\n\nThe longest stage. It saves a checkpoint every 50 steps. "
         "To skip it, run `!python -m capstone.fetch --what sft $A` instead: it merges our SFT adapter into the base."),
    (CODE, "!python -m capstone.sft $A"),
    (MD, "## Stage 3. On-policy samples, then reward-model labels and pairs"),
    (CODE, "!GEN_BATCH=32 python -m capstone.sample $A\n!python -m capstone.label $A"),
    (MD, "## Stage 4. DPO with QLoRA"),
    (CODE, "!python -m capstone.dpo $A"),
    (MD, "## Stage 5. Evaluation generations, IFEval and XSTest scores, then the judge\n\n"
         "The T2 judge is Olmo-3-7B-Instruct in 4-bit. Our run checked it against HelpSteer3 human labels; the page gives "
         "its agreement next to the T3 judge's. On a T4 the default judge cell runs three comparisons and takes about 7 hours "
         "(calibration 34 min, then 129 min per comparison). Add `--pairs dpo:official` to run one comparison in about 163 minutes."),
    (CODE, "!GEN_BATCH=16 python -m capstone.evalgen $A\n!python -m capstone.scores $A\n!GEN_BATCH=4 python -m capstone.judge $A\n"
           "# a second refusal rule, checked against XSTest's human labels first\n!GEN_BATCH=8 python -m capstone.refusal $A"),
    (MD, "## Stage 5d. Label 40 of your own comparisons by hand\n\nExport, then open `handlabel/sheet.jsonl`, read each "
         "request and both replies, and write A, B or C in `label` before you look at any judge verdict. Then score."),
    (CODE, "!python -m capstone.handlabel export $A --n 40\n# ... fill the labels, then:\n# !python -m capstone.handlabel score $A"),
    (MD, "## Stage 6. Model card, then release to your own Hub account\n\nWrite `limitations.md` from your results first."),
    (CODE, "!python -m capstone.card $A --limitations limitations.md\n"
           "# from huggingface_hub import HfApi; HfApi().upload_folder(folder_path=f'{WORK}/dpo/merged', repo_id='<you>/<name>')"),
]

nb = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                "accelerator": "GPU", "colab": {"gpuType": "T4"}}, "nbformat": 4, "nbformat_minor": 5}
for i, (kind, src) in enumerate(cells):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": "c%02d" % i}
    if kind == CODE:
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lltCapstoneAssistant_T2.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
