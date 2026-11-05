# lltBaseVsInstruct: Base vs Instruct, What Post-Training Changes

LLM Training Lab, module 10. Regime: INFERENCE & EVAL. No weights change; you compare SmolLM2-360M with SmolLM2-360M-Instruct token by token.

Open the notebook in Colab (CPU runtime is enough):
https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/courses/llm-training/labs/lltBaseVsInstruct/lltBaseVsInstruct.ipynb

## Files

| File | What it is |
|---|---|
| `lltBaseVsInstruct.ipynb` | The notebook. Its cells are the lab page's code cells, in order, after a runtime check, the pinned install and a cell that writes `prompts.py`. |
| `prompts.py` | The 40 prompts, five groups of 8 (how-to, explanations, code, writing, maths), written for this lab. |
| `steps/` | The page cells as files (`cellNN_stepK.py`) and the challenge starter and solution. Written by `build_notebook.mjs`. |
| `build_notebook.mjs` | Builds the notebook and `steps/` from the module file, so the page and the notebook cannot disagree. `node build_notebook.mjs <path to lltBaseVsInstruct.mjs>`. |
| `measure.py` | Writes every Run Library file in `../../runs/lltBaseVsInstruct/`. Commands: `side`, `template`, `wrong`, `kl smollm2`, `kl qwen3`, `broken` (after `kl smollm2`), `checkpoints`. |
| `run_steps.py` | Runs every page cell in order and records the output (`lltBaseVsInstruct-t1-cells`). |
| `measure_timing.py` | The page's timing figures, run on an idle machine. Prints one JSON line. |
| `runlog.py` | Writes one Run Library JSON file (plan section 7.6 schema). |

## Models and licences

| Model | Use | Licence |
|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` and `HuggingFaceTB/SmolLM2-360M-Instruct` | The base and instruct pair (steps 1 to 6) | Apache-2.0 |
| `Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen3-0.6B` | The second pair (open question, `kl qwen3`) | Apache-2.0 |

## Reproduce

```
pip install torch transformers==5.22.1 tokenizers==0.23.3 huggingface_hub==2.4.0 numpy psutil
python measure.py side
python measure.py template
python measure.py wrong
python measure.py kl smollm2
python measure.py broken
python measure.py kl qwen3
python measure.py checkpoints
node build_notebook.mjs <path to lltBaseVsInstruct.mjs>
python run_steps.py
```

Recorded on a laptop CPU (AMD Ryzen 9 5900HS, 16 threads), float32, greedy decoding, no GPU. The page quotes the `-v2026-11` run files, measured with Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, tokenizers 0.23.3, huggingface_hub 2.4.0 and numpy 2.5.4 (set `LLT_RUN_SUFFIX=-v2026-11` to write them). The files without the suffix are the first runs, on torch 2.10.0, transformers 5.2.0 and tokenizers 0.22.1. `LLT_THREADS` sets the torch thread count (default 4). Wall-clock in the run files was measured while other jobs shared the CPU, so the page takes its times from `measure_timing.py` only.

## Future run

`measure.py kl` stores, per answer token, the KL, the base rank of the chosen token and its category, but not each model's top-k tokens and probabilities. A widget that shows both distributions at a token needs a future run that also stores the top 5 tokens of each model at every position.
