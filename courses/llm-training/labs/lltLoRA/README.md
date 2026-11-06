# lltLoRA: LoRA From First Principles, Measured Against Full Fine-Tuning

LLM Training Lab, module 12. Base model for every GPU step: `Qwen/Qwen3.5-0.8B-Base` (Apache-2.0), task data GSM8K (`openai/gsm8k`, MIT).

## Files

| File | What it is |
|---|---|
| `lltLoRA.ipynb` | The learner notebook, built by `build_notebook.py` from `steps/`. Runs end to end on one free T4 |
| `steps/check_runtime.py` | First cell: Python, GPU, RAM and disk checks |
| `steps/s01_lora_linear.py` to `s04_peft_check.py` | Steps 1 to 4 (CPU or GPU): `LoRALinear`, merge, injection into the lab GPT, the PEFT check |
| `steps/gpt_model.py` | The GPT from lltGPT (steps 2 to 4 of that lab), definitions only |
| `steps/lora_lab.py` | Data, training loop and evaluation shared by the notebook and our Kaggle runs |
| `steps/constants.py`, `s05_full_ft.py` to `s08_sweep.py` | Steps 5 to 8 (T4): full fine-tune, SVD of its update, LoRA, your own sweep |
| `steps/challenge_starter.py`, `challenge_solution.py` | The in-page NumPy challenge |
| `run_steps.py` | Runs steps 1 to 4 on CPU and writes `runs/lltLoRA/lltLoRA-t1-cells-v2026-11.json` |
| `pyodide_check.mjs` | Runs the challenge in Pyodide 0.27 and writes `runs/lltLoRA/lltLoRA-t0-pyodide.json` |
| `measure_timing.py` | CPU timing keys for the page, run on an idle machine |
| `make_runs.py` | Converts the Kaggle results in `kaggle/out/llt-lora-*` into Run Library files |

Our recorded runs are in `../../runs/lltLoRA/`. The Kaggle kernels are built from `../../kaggle/cells_lora_*.py` and `../../kaggle/lora_worker.py`.
