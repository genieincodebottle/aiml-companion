# lltGPT, Assemble a Modern GPT

LLM Training Lab, module 4. Regime: FROM SCRATCH (step 8 trains six small variants from random init; step 7 loads OpenAI's GPT-2 weights and changes nothing).
Every number on the lab page comes from a run in `../../runs/lltGPT/` or from a derivation. No GPU is needed.

The model matches our Kaggle TinyStories pretraining test (`../../kaggle/cells_tinystories.py`) as the **modern** config: pre-norm RMSNorm, RoPE on interleaved pairs, SwiGLU with the hidden width rounded up to a multiple of 64, tied embeddings, init std 0.02 and 0.02/sqrt(2L) on the two residual writes per block. The **gpt2** config (LayerNorm, tanh GELU, learned positions, biases) loads OpenAI's GPT-2 124M weights.

## Run it on a laptop

Use Python 3.11 or later (3.12 recommended). In a fresh virtual environment:

```
pip install torch numpy jupyter ipywidgets
jupyter notebook lltGPT.ipynb
```

Run the cells in order. The notebook installs the pinned Hugging Face libraries itself and keeps your PyTorch and NumPy. Step 7 downloads GPT-2 (about 0.55 GB) and step 8 a 22 MB TinyStories file.

## Files

| File | What it does |
|---|---|
| `lltGPT.ipynb` | The lab notebook, built from `steps/` by `build_notebook.py`. Runs every step on a CPU. |
| `steps/step1.py` ... `steps/step8b.py` | The page's code cells, in page order. Steps 2 to 8b share one Python namespace. `step1.py` and the challenge files are NumPy and run in the browser. |
| `steps/challenge_starter.py`, `steps/challenge_solution.py` | The Prove it challenge (parameter count). |
| `measure.py` | Reproduces every exact figure on the page (runs the three scripts below). |
| `measure_steps.py` | Runs every cell, records its printed output and peak RAM -> `lltGPT-t1-steps`. |
| `measure_gpt2.py` | GPT-2 124M logit match, HF's own SDPA vs eager gap, float64 reference, and the loading-mistake table -> `lltGPT-t1-gpt2-match`, `lltGPT-t1-broken-conv1d-square`. |
| `measure_toggles.py` | Step 8 on a CPU. Our CPU reference run is `lltGPT-t1-toggle-modern-s0` (2.5973). |
| `../../kaggle/cells_gpt_toggles.py`, `cells_gpt_lr.py` | Step 8 table on a Kaggle T4 in float32 (same cells, tensors moved to the GPU), one worker per GPU -> `lltGPT-t2-toggle-*`, `lltGPT-t2-broken-*`, `lltGPT-t2-lr-*`. Kernels `llt-gpt-toggles`, `llt-gpt-lr`. |
| `check_unigram.py` | Loss of predicting each token by its training frequency -> `lltGPT-t1-unigram`. |
| `check_tolerance.py` | The step 7 allclose check against both HF attention paths and the two small mistakes. |
| `summarize_toggles.py` | Prints the step 8 table and curves (`t2` or `t1`). |
| `measure_timing.py` | Every timing figure on the page. Run it on an idle machine; it prints one JSON line. |
| `pyodide_check.mjs` | Runs the browser cells in real Pyodide 0.27.0 -> `lltGPT-t0-pyodide`. |
| `run_steps.py` | Runs chosen step cells in one namespace and prints their output. |
| `runlog.py` | Writes one Run Library JSON per run (plan section 7.6 schema). |

## Reproduce the Run Library

```
python measure_steps.py
python measure_gpt2.py
python measure_toggles.py --seeds 0 --only modern      # CPU reference point
# step 8 table: build and push kaggle/llt-gpt-toggles and llt-gpt-lr (build_nb.py), then copy out/*/runs/*.json here
python check_unigram.py
python check_tolerance.py
python measure_timing.py          # idle machine only
python build_notebook.py
```

Exact figures were measured with `torch.set_num_threads(4)`. The step 8 losses can move in the fourth decimal with another thread count.

Data: `roneneldan/TinyStories`, file `TinyStoriesV2-GPT4-valid.txt` (CDLA-Sharing-1.0, Eldan and Li 2023), 27,630 stories split 95/5 in file order; a 2,048-token byte-level BPE trained on the 95%.
Model: `openai-community/gpt2` (MIT licence).

Environment of the CPU runs on the page (suffix `-v2026-11`): the venv `../../.venv-latest`, Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, tokenizers 0.23.3, huggingface_hub 2.4.0, numpy 2.5.4, AMD Ryzen 9 5900HS, no GPU. Run the measure scripts with `LLT_RUN_SUFFIX=-v2026-11` and that interpreter. Every figure matched the earlier torch 2.10.0 runs except peak RAM (2.97 -> 2.87 GB). The CPU reference `lltGPT-t1-toggle-modern-s0` (2.5973) ran on torch 2.10.0+cpu, tokenizers 0.22.1, and gave 2.5973 again on torch 2.15.0+cpu, tokenizers 0.23.3, transformers 5.22.1 (`lltGPT-t1-toggle-modern-s0-v2026-11b`, 2026-11-06). The step 8 table ran on the Kaggle T4 image of 2026-11-05: Python 3.13.16, torch 2.11.0+cu130, tokenizers 0.22.1. Browser cells checked on Pyodide 0.27.0.
