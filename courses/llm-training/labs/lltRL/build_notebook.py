"""Build the learner notebook lltRL.ipynb from steps/cellNN_stepK.py.

The page's code blocks come from the same files (export_page_code.py), and the Google Cloud job runner
(../../gcp/llt-rl/run_job.py) executes the same files with only flags and sizes patched, so page, notebook and
recorded runs cannot disagree. After building, run ../../tools/add_run_help.py (the "How to run this notebook" cell).
  python build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "Step 1. Derive the policy gradient and write REINFORCE with a baseline (NumPy, CPU)",
         2: "Step 2. Write the PPO pieces and test them (PyTorch, CPU)",
         3: "Step 3. Load the four PPO models and score a rollout (GPU from here)",
         4: "Step 4. Train PPO on IMDb prefixes (RUN_PPO = False downloads our policy)",
         5: "Step 5. Sweep the KL coefficient (RUN_EXPERIMENT = True trains two more policies)",
         6: "Step 6. Remove the clip, cut the epochs (RUN_EXPERIMENT = True trains two more policies)",
         7: "Step 7. Write a verifiable reward and group advantages (CPU)",
         8: "Step 8. Write the GRPO loss and match GRPOTrainer on one batch (CPU)",
         9: "Step 9. Load the GRPO policy and measure the base model",
         10: "Step 10. Train GRPO by hand (START_FROM = None trains all 150 steps)",
         11: "Step 11. Measure the trained policy",
         12: "Step 12. Change the group size, the KL term, the normalization and the baseline (RUN_EXPERIMENT = True)",
         13: "Step 13. Run the same training with GRPOTrainer (RUN_TRL = True)"}

INTRO = ("# RL for LLMs: PPO from scratch, then GRPO\n\nLLM Training Lab. Part 1 trains SmolLM2-135M with a hand-written PPO "
         "to continue IMDb review openings positively, scored by a sentiment classifier. Part 2 trains a LoRA adapter on "
         "Qwen3-0.6B-Base with GRPO on GSM8K, scored by a verifiable reward.\n\n"
         "Run the cells in order on one T4 (Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > "
         "GPU T4 x2). Steps 1 and 2 also run on a CPU. Code cell numbers match the lab page.")

RUNTIME = (HERE.parent / "lltRewardModels" / "build_notebook.py").read_text(encoding="utf-8").split("RUNTIME = '''")[1].split("'''")[0]
RUNTIME = RUNTIME.replace("steps 1 to 4 still run. For step 5 on", "steps 1, 2, 7 and 8 still run. For step 3 on").replace("< 6", "< 8").replace("at least 6 GB", "at least 8 GB")
assert "steps 1, 2, 7 and 8 still run" in RUNTIME

INSTALL = '''# Pinned Hugging Face stack of our recorded runs. torch and numpy stay as the runtime ships them.
%pip uninstall -y -q torchao
%pip install -q transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 tokenizers==0.23.3 huggingface_hub==2.4.0 math-verify==0.9.0'''


def nb(cells):
    return {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5}


cells = [("md", INTRO), ("code", RUNTIME), ("code", INSTALL)]
seen = set()
for i, path in enumerate(sorted((HERE / "steps").glob("cell*_step*.py")), start=1):
    step = int(path.stem.split("_step")[1])
    if step not in seen:
        seen.add(step)
        cells.append(("md", "## " + STEPS[step] + "\n\nCode cell " + str(i) + " on the lab page."))
    cells.append(("code", path.read_text(encoding="utf-8").rstrip()))
with open(HERE / "lltRL.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb(cells), f, indent=1)
print("learner cells", len(cells))
