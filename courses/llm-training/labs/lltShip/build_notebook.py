"""Build the learner notebook lltShip.ipynb from steps/cellNN_stepK.py, the same files the page's code blocks come from
(export_page_code.py), so page and notebook cannot disagree. After a rebuild, run tools/add_run_help.py.

    python build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NL = chr(10)
STEPS = {1: "Step 1. Check that your fine-tunes share a base (CPU)",
         2: "Step 2. Compute the task vectors and look at them (CPU)",
         3: "Step 3. Write linear, task arithmetic, TIES and DARE (CPU)",
         4: "Step 4. Build a two-skill panel and score the parents (GPU)",
         5: "Step 5. Merge, score both skills, pick one (FROM_OUR_CHECKPOINT scores our merge, RUN_GRID = True scores 9 more)",
         6: "Step 6. Reproduce your TIES merge with mergekit (CPU)",
         7: "Step 7. Convert to GGUF and compare k-quants (llama.cpp, CPU; FROM_OUR_CHECKPOINT loads our files)",
         8: "Step 8. Quantize to 4 bits with GPTQ, with two calibration sets (GPU)",
         9: "Step 9. When rounding fails, train with fake quantization (QAT, GPU; FROM_OUR_CHECKPOINT loads our result)",
         10: "Step 10. Export with the right config, chat template and model card",
         11: "Step 11. Serve with llama.cpp and run a smoke eval through the endpoint",
         12: "Step 12. Serve with vLLM, one merged model or one base with adapters (RUN_VLLM = True; FROM_OUR_CHECKPOINT prints our run)"}

INTRO = ("# From checkpoint to endpoint: merge, compress, serve" + NL + NL +
         "LLM Training Lab, module 22. Regime POST-HOC: no new fine-tuning. You merge two of your earlier fine-tunes of "
         "Qwen3-0.6B-Base, compress the merge three ways, and serve it." + NL + NL +
         "Steps 1 to 3 and 6 run on a CPU. Steps 4, 5, 8 and 9 need a GPU (Colab: Runtime > Change runtime type > T4 GPU. "
         "Kaggle: Settings > Accelerator > GPU T4 x2, Internet on). Steps 7 and 11 run llama.cpp on the CPU of the same "
         "session. Code cell numbers match the lab page." + NL + NL +
         "**Free T4 path: `FROM_OUR_CHECKPOINT = True`, set near the top of code cell 1, and the default.** The long steps then use our "
         "recorded results from the Hugging Face Hub instead of running again: step 5 scores our merge, step 7 loads our GGUF file, "
         "step 9 prints our QAT run and step 12 prints our vLLM run. Every later step still runs on them. Set it to `False` to run "
         "every step yourself, which took 75.5 min on our T4 VM against 34.0 min. Steps 5 and 12 have two more switches, `RUN_GRID` and `RUN_VLLM`.")

RUNTIME = (HERE.parent / "lltRewardModels" / "build_notebook.py").read_text(encoding="utf-8").split("RUNTIME = '''")[1].split("'''")[0]
RUNTIME = RUNTIME.replace("steps 1 to 4 still run. For step 5 on", "steps 1 to 3 and 6 still run. For steps 4, 5, 8 and 9")
RUNTIME = RUNTIME.replace("< 6", "< 20").replace("at least 6 GB", "at least 20 GB")

INSTALL = '''# Pinned stack of our recorded run. torch and numpy stay as the runtime ships them.
%pip uninstall -y -q torchao
%pip install -q transformers==5.22.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0 mergekit==0.1.5 gguf==0.19.0'''


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


cells = [("md", INTRO), ("code", RUNTIME), ("code", INSTALL)]
for i, path in enumerate(sorted((HERE / "steps").glob("cell*_step*.py")), start=1):
    step = int(path.stem.split("_step")[1])
    cells.append(("md", "## " + STEPS[step] + NL + NL + "Code cell " + str(i) + " on the lab page."))
    cells.append(("code", path.read_text(encoding="utf-8").rstrip()))
with open(HERE / "lltShip.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb(cells), f, indent=1)
print("cells", len(cells))
