"""Build lltGeneration.ipynb from the step files, so the notebook and the page run the same code.

    python build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def src(*parts):
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read().rstrip() + "\n"


CELLS = [
    ("md", "# LLM Training Lab, lab 6. Generation and the KV cache, implemented\n\n"
           "INFERENCE & EVAL. No weights change in the lab steps. You pretrain a small GPT on a CPU first, then sample from it, "
           "add a KV cache with a RoPE position offset, prove the cached logits match, count the cache bytes and time it.\n\n"
           "Runs on a CPU. Steps follow the lab page in this app; run the cells in order."),
    ("code", '%pip install -q "transformers==5.22.1" "tokenizers==0.23.3" "huggingface_hub==2.4.0"' + chr(10)),
    ("md", "If the install cell changed a package that was already loaded, use Runtime > Restart session, then run the check below."),
    ("code", src("steps", "nb_check.py")),
    ("md", "## Setup. The model\n\nThe GPT from lltGPT, with RMSNorm, RoPE, SwiGLU and tied embeddings, plus `n_kv_heads` for step 8."),
    ("code", src("steps", "nb_model.py")),
    ("md", "## Setup. Pretrain the lab model on a CPU\n\n2.2M parameters, 1,500 steps of 16 x 256 tokens on TinyStoriesV2-GPT4-valid.txt (CDLA-Sharing-1.0). "
           "Our run lltGeneration-t1-model-s0 went from val loss 7.67 at step 0 to 2.16 at step 1,500."),
    ("code", src("steps", "nb_train.py")),
    ("md", "## Step 1. Temperature, top-k and top-p on eight words\n\nThe same cell runs in the lab page with NumPy."),
    ("code", src("browser", "step1.py")),
    ("md", "## Step 2. Generate without a cache"),
    ("code", src("steps", "step2.py")),
    ("md", "## Step 3. Temperature and top-p on your model"),
    ("code", src("steps", "step3.py")),
    ("md", "## Step 4. Add a preallocated KV cache with a position offset"),
    ("code", src("steps", "step4.py")),
    ("md", "## Step 5. Prove cached and uncached logits match"),
    ("code", src("steps", "step5.py")),
    ("md", "## Step 6. Count the cache bytes\n\nYour model first, then SmolLM2-135M's real Hugging Face cache (downloads about 270 MB)."),
    ("code", src("steps", "step6.py")),
    ("code", src("steps", "step6b.py")),
    ("md", "## Step 7. Time generation with and without the cache"),
    ("code", src("steps", "step7.py")),
    ("md", "## Step 8. Shrink the cache with grouped-query attention"),
    ("code", src("steps", "step8.py")),
    ("md", "## Optional. Reproduce the two broken runs"),
    ("code", src("steps", "nb_broken.py")),
    ("md", "## Optional. Research extension, induction heads\n\nTrains a 2-layer attention-only model on repeated random tokens and prints the induction score "
           "and the ICL score every 50 steps. Watch for the step where both change together."),
    ("code", src("steps", "nb_research.py")),
]


def main():
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
            for k, s in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = os.path.join(HERE, "lltGeneration.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", path, len(CELLS), "cells")


if __name__ == "__main__":
    main()
