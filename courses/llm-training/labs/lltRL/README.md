# lltRL: write PPO from scratch, then train with GRPO

LLM Training Lab, module 19. Regime RL. Part 1 trains every weight of SmolLM2-135M with a hand-written PPO to continue
IMDb review openings positively, scored by the `lvwerra/distilbert-imdb` classifier. Part 2 trains a rank-16 LoRA adapter
on Qwen3-0.6B-Base with GRPO on GSM8K, scored by a 0-or-1 verifiable reward (math-verify).

TRL removed `PPOTrainer` in v1.13.0 (2026-09-10), so PPO here is written by hand. GRPO is written by hand and then
compared with TRL's `GRPOTrainer` (TRL 1.17.1).

## Files

| File | What it is |
|---|---|
| `lltRL.ipynb` | The learner notebook. Built by `build_notebook.py` from `steps/`; every code cell is a code block on the lab page |
| `steps/cellNN_stepK.py` | The page code cells, one file each. Steps 1, 2, 7 and 8 run on a CPU, the others on one T4 |
| `challenge_starter.py`, `challenge_solution.py` | The in-page NumPy challenge (PPO clipped loss and gradient, GRPO advantages) |
| `build_notebook.py` | Builds the notebook; then run `../../tools/add_run_help.py` |
| `export_page_code.py` | Puts the cell files into the page draft as JS constants |
| `run_steps.py`, `measure_timing.py` | Run and time the CPU steps; write `runs/lltRL/lltRL-t1-cells.json` |
| `record_gpu.py` | Turns a GPU job output (`result.json`, `stdout.txt`) into Run Library files in `runs/lltRL/` |
| `upload_checkpoints.py` | Uploads a saved model or adapter to the Hub under `lltRL/` |
| `tryit_data.py` | Builds `tryit_data.json` for the Try it widget from recorded runs |
| `TRYIT_SPEC.md`, `OVERVIEW_SPEC.md` | Specs for the widget and the narrated overview |
| `../../gcp/llt-rl/` | The GPU job runner (`run_job.py` runs the same step files with flags and sizes patched), chain scripts, requirements, env setup |

## Recorded runs

GPU runs ran on a Google Cloud A100 (llt-c3) with the T4 recipe: fp16 autocast with a GradScaler, fp16 frozen bases,
fp32 trained weights. Numerics follow the T4 path; wall-clock and memory are A100 numbers, and the A100 was shared by
several jobs at once. T4 times on the page are projections until the notebook is re-timed on a Kaggle T4.

Checkpoints on the Hugging Face Hub, `aimlcompanion/llm-training-lab`:
- `lltRL/ppo-kl0.05-s0/`: SmolLM2-135M after 200 PPO iterations, KL coefficient 0.05, seed 0.
- `lltRL/grpo-s0-step130/`: the GRPO LoRA adapter after 130 of 150 steps (the notebook's default start).
- `lltRL/grpo-s0-step150/`: the same run after all 150 steps.

## Licences

IMDb (`stanfordnlp/imdb`) carries the licence field "other" (released for research use); this lab uses its openings
only as prompts. `lvwerra/distilbert-imdb`, SmolLM2-135M and Qwen3-0.6B-Base are Apache-2.0. GSM8K is MIT.
