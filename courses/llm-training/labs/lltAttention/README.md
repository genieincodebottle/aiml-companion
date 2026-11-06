# lltAttention, Causal Self-Attention, Built and Tested

LLM Training Lab, module 3. Regime: FROM SCRATCH for the small character model in step 8. Steps 1 to 7 train nothing; they build, test and measure one attention layer.

Open the notebook in Colab (a CPU runtime runs every step except step 7, which needs a T4):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltAttention/lltAttention.ipynb

## Run it on a laptop

Use Python 3.11 or later (3.12 recommended). In a fresh virtual environment:

```
pip install torch numpy jupyter
jupyter notebook lltAttention.ipynb
```

Run the cells in order. Step 7 needs a CUDA GPU; on a CPU it prints one line and stops. Step 8 downloads 22.5 MB from the Hugging Face Hub.

## Files

| File | What it is |
|---|---|
| `lltAttention.ipynb` | The notebook. A runtime check, then the lab page's code cells in order. Built by `build_notebook.py`. |
| `cells/step1.py` to `step3.py` | Browser cells (NumPy only, 50 lines or fewer). They run in the page on Pyodide 0.27. |
| `cells/step4.py` to `step6.py` | PyTorch module, SDPA match, leakage test, CPU peak memory. One shared namespace, like notebook cells. |
| `cells/step_t4.py` | Page step 7, naive attention vs SDPA on a CUDA GPU. |
| `cells/step7.py` | Page step 8, three 600-step training runs with a correct, a missing and a misplaced mask. |
| `cells/challenge_starter.py`, `challenge_solution.py` | The Prove it challenge. The starter prints `All checks passed` once the four functions are right. |
| `measure.py` | Regenerates every CPU Run Library file in `../../runs/lltAttention/`. `python measure.py` runs all sections. |
| `import_kaggle.py` | Turns the Kaggle T4 output into Run Library files. |
| `measure_timing.py` | Laptop timing figures for the page, one JSON line. Run it on an idle machine. |
| `pyodide_time.mjs` | Times the browser cells in real Pyodide 0.27.0 under Node. |
| `run_steps.py` | Runs page cells in one namespace and prints their output. |
| `runlog.py` | Writes one Run Library JSON per run. |

## GPU runs

The T4 figures come from two private Kaggle kernels built from `../../kaggle/`:

- `llt-attention-t4` (`cells_attention_t4.py`): every SDPA backend for T = 128 to 4,096, float16 accuracy, a float16 leakage test and the -1e9 mask checks.
- `llt-attention-t4-cell` (`cells_attention_t4_cell.py`): the page's step 7 cell run exactly as shipped.

Both ran on one Tesla T4 (capability 7.5), Python 3.13, torch 2.11.0+cu130, on 2026-11-05. The page's step 7 cell was re-run on torch 2.15.0+cu132 the same day (`lltAttention-t2-t4-cell-v2026-11`).

## Data

Step 8 downloads `TinyStoriesV2-GPT4-valid.txt` (22.5 MB) from `roneneldan/TinyStories` on the Hugging Face Hub (Eldan and Li 2023, CDLA-Sharing-1.0) and uses its first 1,000,000 characters, 900,000 for training and 100,000 for validation.

## Reproduce

```
pip install torch==2.15.0 numpy psutil nbformat
python measure.py                     # all CPU runs
python import_kaggle.py               # after downloading the Kaggle outputs to ../../kaggle/out/
python measure_timing.py              # idle machine only
python build_notebook.py
```

Recorded on a laptop CPU (AMD Ryzen 9 5900HS, 16 threads), Python 3.12.3, torch 2.15.0+cpu, numpy 2.5.4, with `torch.set_num_threads(4)`. Memory figures, SDPA matches and leakage results repeat exactly across torch 2.10.0 and 2.15.0. Training losses repeat to 5 decimals except the mask-after-softmax run, where the last-batch loss moved from 0.034 to 0.035 and the prefix-only validation loss from 7.138 to 7.134 (`-v2026-11` run files). Times depend on the machine and its load.
