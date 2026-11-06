# lltQLoRA: Fine-Tune a 9B Model on a Free GPU With QLoRA

LLM Training Lab, module 13. Regime PEFT: Qwen3.5-9B-Base frozen in 4-bit NF4, LoRA adapters on all 248 linear layers,
objective masked NLL on the answer tokens.

Open the notebook in Colab (Runtime > Change runtime type > T4 GPU):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltQLoRA/lltQLoRA.ipynb

The full notebook took 3 h 55 min on one Kaggle T4 (steps 1 to 8: 37 min). Steps 10 to 12 are long: step 10 can load our
Optuna journal (`USE_OUR_STUDY = True`), step 11 can run one seed, and step 12 is optional.

## Files

| File | What it is |
|---|---|
| `lltQLoRA.ipynb` | The learner notebook, built from `cells_lab.py` by `build_notebook.py`. Its code cells are the lab page's code blocks, numbered the same. |
| `cells_lab.py` | The cells. Kinds: `code` (shipped), `md`, `rec` (Kaggle measurement kernel only: records results to JSON). |
| `build_notebook.py` | Writes `lltQLoRA.ipynb` and the Kaggle kernel `../../kaggle/llt-qlora-lab/`. |
| `export_page_code.py` | Writes the page's code constants from `cells_lab.py`. |
| `fetch_weights.py` | Reads real Qwen3.5-9B-Base tensors with HTTP range requests (no 19 GB download). |
| `measure.py` | CPU measurements: `nf4` (levels, NF4 vs int4 vs FP4, block sweep, bitsandbytes match, double quant), `mem` (parameter counts and the predicted NF4 footprint), `tryit` (the Try it JSON). |
| `make_runs.py` | Writes the Run Library files from the Kaggle output and earlier runs. |

## Run Library (`../../runs/lltQLoRA/`)

| Run | What |
|---|---|
| `lltQLoRA-T2-lab-v2026-11` | The notebook end to end on a Kaggle T4 (kernel `llt-qlora-lab` v3, 2026-11-06): load, memory sweep, base and fine-tuned accuracy, loss check, resume, Optuna study, three arms, tuned 9B. |
| `lltQLoRA-T2-lab-run2-partial-v2026-11` | Kernel v2: steps 1 to 9 again (stopped in step 10 by an out-of-memory error, since fixed). |
| `tune.log` | The Optuna 5.0 journal of the 8-trial study, which `USE_OUR_STUDY = True` downloads. |
| `lltQLoRA-t1-nf4-v2026-11`, `lltQLoRA-t1-memory-v2026-11` | CPU measurements from `measure.py`. |
| `lltQLoRA-T2-broken-*` | Diagnose-it runs, taken from earlier Kaggle runs (`llt-model-probe`, `llt-finetune-test`). |

## Data

`PolyAI/banking77` (CC-BY-4.0): 10,003 training and 3,080 test messages, 77 intents. The lab trains on 3,000 training
messages, tunes on 500 held-out training messages and tests on all 3,080 test messages.

## Reproduce the CPU figures

```
python fetch_weights.py <weights_dir>
python measure.py nf4 <weights_dir>
python measure.py mem <weights_dir>
python measure.py tryit <weights_dir> tryit.json
```
Needs numpy, scipy, torch, requests and bitsandbytes 0.50.2 (its CPU backend).
