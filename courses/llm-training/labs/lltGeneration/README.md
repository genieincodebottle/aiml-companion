# lltGeneration, Generation and the KV Cache, Implemented

LLM Training Lab, module 6. Regime: INFERENCE & EVAL. No weights change in steps 1 to 8; the setup cell
pretrains the lab model FROM SCRATCH on a CPU. Every number on the lab page comes from a run in
`../../runs/lltGeneration/` or from an exact derivation.

## Files

| File | What it does |
|---|---|
| `lltGeneration.ipynb` | The lab notebook, built by `build_notebook.py` from `steps/` and `browser/`. Runs on a CPU. |
| `model.py` | The GPT of `kaggle/cells_tinystories.py` plus GQA (`n_kv_heads`), a preallocated `KVCache` and a RoPE position offset. `bug` reproduces the two broken runs. `mlp=False` gives an attention-only model. |
| `train_small.py` | Pretrains the lab model (2,048-token BPE, d 192, 4 layers, 6 heads, ctx 256) on TinyStoriesV2-GPT4-valid.txt. Writes `out/` and run `lltGeneration-t1-model-s0`. |
| `measure.py` | Exact measurements through `model.py`: sampling, cache exactness, KV bytes (incl. SmolLM2-135M's real cache), GQA, the broken runs, help-panel errors. |
| `run_steps.py` | Runs the page's code blocks (`steps/`) in one namespace, as the notebook does, and records their output. |
| `measure_timing.py` | Every timing figure on the page. Run on an idle machine; prints one JSON line. |
| `induction.py` | Research extension. Induction score and ICL score; trains a 2-layer attention-only model and a 1-layer control on CPU. |
| `ts_induction.py`, `ts_repeat_text.py` | The same scores on the lltTinyStories baseline checkpoints in `../../kaggle/out/llt-tinystories-pretrain-test/ckpt/` (steps 0, 1,000, 2,000, 3,000). |
| `ts_sweep_induction.py` | The same scores on the final checkpoints of the lltTinyStories sweep (3M, 11M seed 1, deep, wide, 31M) in `../../kaggle/out/llt-tinystories-sweep/`. |
| `run_research_cell.py` | Runs the notebook's research cell and records its output. |
| `browser/` | Step 1 and the Prove it challenge, NumPy only (they run in the lab page on Pyodide). |
| `steps/` | The notebook cells, one file per page step. |
| `build_notebook.py` | Builds `lltGeneration.ipynb` from `steps/` and `browser/`. |
| `runlog.py` | Writes one Run Library JSON per run. |
| `run_all.sh`, `run_rest.sh` | Re-run every exact measurement in `.venv-latest`, one job at a time (`run_rest.sh` skips training). |
| `measure_qwen_params.py` | Counts Qwen3-0.6B's distinct parameters (tied embedding once) and bf16 bytes for interview answer 2. |
| `verify_nb_train.py` | Checks that the notebook's training cell reproduces `train_small.py` (val loss at step 250). |
| `browser/record_pyodide.py` | Records the in-page cells' output under Pyodide 0.27.0 next to CPython. |

## Reproduce the Run Library

```
python train_small.py --seed 0 --threads 4
python measure.py --threads 4
python run_steps.py
python induction.py --seed 0 --threads 2 --steps 1500 --eval-every 25
python induction.py --seed 0 --threads 2 --steps 1500 --eval-every 25 --layers 1
python induction.py --seed 0 --threads 2 --steps 1000 --eval-every 25 --layers 1 --fixed-period
python run_research_cell.py --threads 2
python ts_induction.py --threads 4
python ts_repeat_text.py --threads 4
python ts_sweep_induction.py --threads 2
node <pyodide-test>/run_pyodide.mjs browser/step1.py > out_pyo1.txt   # and challenge_solution.py > out_pyo2.txt
python browser/record_pyodide.py out_pyo1.txt out_pyo2.txt
python build_notebook.py
python measure_timing.py
```

`out/` holds the trained checkpoint and tokenizer (not committed).

Data: `roneneldan/TinyStories`, file `TinyStoriesV2-GPT4-valid.txt` (CDLA-Sharing-1.0, Eldan and Li 2023), split 95/5 by story with shuffle seed 0; the TinyStories `validation` split for the checkpoint scores.
Models: SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`, Apache 2.0) for the cache-bytes check.

Environment of the recorded T1 runs: `.venv-latest`, Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, tokenizers 0.23.3, huggingface_hub 2.4.0, datasets 5.1.0, numpy 2.5.4, AMD Ryzen 9 5900HS, no GPU, `torch.set_num_threads(4)`. Run every script with `../../.venv-latest/Scripts/python.exe`. The in-page cells were checked on Pyodide 0.27.0.
