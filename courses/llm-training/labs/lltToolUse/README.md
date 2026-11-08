# lltToolUse: Fine-Tuning for Tool Calling and Structured Output

LLM Training Lab, module 16. Regime: PEFT. Base model Qwen/Qwen3.5-2B-Base (frozen in fp16) with rank-16 LoRA adapters
in fp32 on every linear layer. Reference arm: Qwen/Qwen3-1.7B-Base, the previous Qwen generation, in its Hermes JSON format.

## Files

| File | What it is |
|---|---|
| `lltToolUse.ipynb` | The learner notebook, built from the lab page by `build_notebook.mjs`. Steps 1 to 6 run on a CPU, steps 7 to 11 need one T4. |
| `tool_common.py` | Shared helpers the notebook writes in its third cell: Hermes conversion, row selection, label mask, BFCL loader, parsers, AST check, constrained decoder (`CallsOnly`), generation loop, scoring, paired bootstrap, `train_lora`. |
| `steps/` | Each page code block as a file (`cellNN_stepK.py`), plus the Prove it starter and solution. |
| `build_notebook.mjs` | `node build_notebook.mjs <path to lltToolUse.mjs>`. Fails if a function the page shows differs from `tool_common.py`. |
| `measure.py` | CPU measurements for the page (T1): `template`, `data`, `roundtrip`, `maskloss`, `specialtokens`, `gencheck`. Writes `runs/lltToolUse/lltToolUse-t1-*.json`. |
| `run_steps.py` | Runs the page's CPU cells in one namespace, as the notebook does. |
| `measure_timing.py` | CPU timing for the quiet-machine pass: prints `t1_steps_min`, `t1_check_min`, `t1_peak_gb` as one JSON line. |
| `make_runs.py` | Turns the Kaggle outputs into Run Library files and the Try it data. |
| `poll_kernel.py` | Waits for a Kaggle kernel and downloads its outputs. |

Kaggle kernels (private) are built by `../../kaggle/make_tooluse_kernels.py`: `llt-tooluse-smoke` (toy sizes),
`llt-tooluse-notebook` (the notebook end to end, the seed-0 run), `llt-tooluse-runs` (Qwen3-1.7B-Base base and LoRA,
the mixed set with 200 negatives, seed 1, the raw Hermes text run) `llt-tooluse-q3tok` (Qwen3 with special-token
rows set before training) and `llt-tooluse-final` (`make_tooluse_final.py`: the final notebook end to end with steps 10 and
11 on, then step 8 with `USE_OUR_ADAPTER = True`; run file `lltToolUse-T2-notebook-final`, written by `make_final_run.py`). Run files: `../../runs/lltToolUse/`. Our seed-0 adapter is on the Hub at
`aimlcompanion/llm-training-lab`, folder `lltToolUse/qwen3.5-2b-lora-s0` (`upload_adapter.py`).

## Data and licences

- `NousResearch/hermes-function-calling-v1` (Apache-2.0): `func-calling-singleturn.json` and `glaive-function-calling-5k.json`,
  the first 600 rows of each whose calls parse and whose rendering fits 1,536 tokens.
- BFCL V4 data from `bfcl-eval==2026.3.23` (Apache-2.0), installed with `--no-deps` because its requirements pin numpy 1.26.4.
  Every 4th item of simple_python, multiple, parallel, parallel_multiple and irrelevance: 310 items.

## Versions

T2: Kaggle image 2026-11-07, Python 3.13.16, torch 2.11.0+cu130, transformers 5.22.1, trl 1.17.1, peft 0.21.2,
accelerate 1.15.0, datasets 5.1.0, one Tesla T4. T1: Python 3.12.3, torch 2.15.0+cpu, transformers 5.22.1, trl 1.17.1,
tokenizers 0.23.3, datasets 5.1.0, numpy 2.5.4.
