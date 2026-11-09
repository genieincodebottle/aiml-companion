# lltDistillation: distillation for LLMs, off-policy, on-policy and logits

LLM Training Lab, module 20. Teacher Qwen3-4B-Base (4-bit NF4, frozen), student Qwen3-0.6B-Base (frozen fp16 base, fp32 LoRA r16 on every linear layer). Both share one tokenizer of 151,936 vocabulary rows, which token-level distillation needs. Task: GSM8K, `Question: ... Answer:` prompts, greedy answers, 500 test questions.

## Files

| File | What it is |
|---|---|
| `lltDistillation.ipynb` | The learner notebook. Built by `build_notebook.py` from `steps/` |
| `steps/cellNN_stepK.py` | One file per code cell. The lab page's code blocks are these files (`export_page_code.py`) |
| `build_notebook.py` | Builds the learner notebook and three Kaggle kernels (smoke, Run Library main, learner defaults) by exact flag replacement |
| `export_page_code.py` | Writes the page draft's code constants from `steps/` and the challenge files |
| `run_steps.py` | Runs the CPU cells (steps 1 and 2) and writes `runs/lltDistillation/lltDistillation-t1-cells.json` |
| `measure_timing.py` | CPU timing of steps 1 and 2 on a quiet machine (one JSON line) |
| `record_gpu.py` | Turns a kernel's `distill_results.json` and `notebook_stdout.txt` into Run Library files |
| `tryit_data.py` | Reference values for the Try it widget |
| `challenge_starter.py`, `challenge_solution.py` | The in-page challenge (NumPy) |

## Flags in the notebook

- `USE_OUR_TEACHER_DATA = True` (step 4) downloads the teacher's 2,048 training answers from `aimlcompanion/llm-training-lab` (`lltDistillation/teacher_train.json`) instead of generating them.
- `USE_OUR_ADAPTER = False` (step 7). `True` loads our on-policy adapter (`lltDistillation/onpolicy-reverse-s0/`) instead of training it.
- `RUN_EXPERIMENT = False` (step 8). `True` trains the other three arms of the grid and a second seed of three arms.

## Hardware

One T4. Checked on Kaggle (Python 3.13, torch 2.11.0+cu130). Colab not run. Times and memory: see the Run Library files.
