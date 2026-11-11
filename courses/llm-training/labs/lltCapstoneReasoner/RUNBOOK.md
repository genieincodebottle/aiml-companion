# Capstone runbook: train a small reasoner with RL (lltCapstoneReasoner)

You train a base Qwen3 model with GRPO on Countdown puzzles, choose the GRPO variant with a 2 x 2 x 2 ablation, and report pass@k, reply length and GPU-hours. This folder is the reference implementation we ran. Every number on the capstone page comes from a run file in `courses/llm-training/runs/lltCapstoneReasoner/`.

## Files

| File | What it does |
|---|---|
| `countdown.py` | Problem generator, prompt, the safe checker and the reward. No `eval()` on model output |
| `make_data.py` | Writes `data/train3.jsonl` (20,000 three-number problems, the training set), `data/train.jsonl` (20,000 with 3 and 4 numbers) and `data/test.jsonl` (200: 100 with 3 numbers, 100 with 4). Same seed, same files |
| `rl_core.py` | Loads the GRPO functions from the lltRL lab's step files by name (advantages, loss, log-probs, sampling, pass@k), so there is one version of them |
| `train.py` | Resumable GRPO loop. Run the same command again after a dropped session and it continues from the last checkpoint |
| `evaluate.py` | pass@k with Hugging Face `generate` (works on a T4) |
| `evaluate_vllm.py` | The same evaluation with vLLM, about ten times faster for many samples (T3 and above, own environment) |
| `analyze.py` | Ablation table, main effects, interaction and seed spread with paired bootstrap intervals |
| `analyze_replies.py` | Operators used, copied wording and true or false arithmetic claims in sampled replies (vLLM) |
| `make_runs.py`, `figures.py` | Run Library files from the run folders, and the page's SVG charts |
| `hf_upload.py` | Our adapters to `aimlcompanion/llm-training-lab/lltCapstoneReasoner/` |
| `test_capstone.py` | CPU tests for the checker, the generator, pass@k and the reused lltRL functions |
| `gcp/` | How we ran the reference on one shared A100 Spot VM (lock file, resumable chain) |

## The ablation

| Factor | Level 1 (original GRPO) | Level 2 (Dr. GRPO) | TRL 1.17.1 equivalent |
|---|---|---|---|
| Loss aggregation `--agg` | `grpo`: mean over each reply's tokens, then over replies | `dr_grpo`: sum over tokens, divided by replies x max length | `loss_type="grpo"` / `"dr_grpo"` |
| Advantage scaling `--scale` | `1`: divide by the group's reward std | `0`: do not | `scale_rewards="group"` / `"none"` |

Two seeds per cell (`--seed 0`, `--seed 1`), 8 runs; we added `--seed 2` (12 runs) after two seeds proved misleading. Everything else is fixed at the `train.py` defaults we used: Qwen3-0.6B-Base, one-shot prompt, reward 1.0 / 0.1 format bonus / 0 (`--reward format`), three-number training problems (`data/train3.jsonl`), 8 problems x 8 replies per step, learning rate 2e-5, 100 steps, 256-token replies, LoRA rank 16, fp16.

## Tier T2: free T4 (Kaggle or Colab), Qwen3-0.6B-Base

1. Runtime: GPU T4. On Kaggle also turn Internet on.
2. Install the pinned stack (do not reinstall torch or numpy):
   `pip install transformers==5.22.1 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 math-verify==0.9.0`
3. Get this folder and the lltRL step files it reuses (sparse clone of the companion repo):
   `git clone --depth 1 --filter=blob:none --sparse https://github.com/genieincodebottle/aiml-companion.git && cd aiml-companion && git sparse-checkout set courses/llm-training/labs/lltCapstoneReasoner courses/llm-training/labs/lltRL && cd courses/llm-training/labs/lltCapstoneReasoner`
4. `python test_capstone.py` (CPU, under a minute).
5. Base model first: `python evaluate.py --out evals/base.json --n 16 --batch 64` (pass@1 to pass@16; ours was pass@16 = 0.17 with vLLM at n = 64).
6. One run of the ablation, stopping cleanly before the session limit:
   `python train.py --out runs/agg-grpo_scale-1_s0 --agg grpo --scale 1 --seed 0 --max_minutes 100`
   Run the same command in the next session to continue. On Kaggle, save `runs/` as a dataset or download it between sessions, since `/kaggle/working` is not kept.
7. Evaluate each finished run: `python evaluate.py --out evals/agg-grpo_scale-1_s0.json --adapter runs/agg-grpo_scale-1_s0/adapter-final --n 16 --batch 64`.
8. After all 8 runs: `python analyze.py`.

Measured on a T4 (2026-11-11, a Google Cloud VM set up like Kaggle's GPU T4 x2, running this notebook top to bottom, one ablation cell per T4):

| Step | Time on one T4 |
|---|---|
| Install cell, tests | 29 seconds, 12 seconds |
| Base model evaluation (`--n 16 --batch 64`) | 38 minutes (includes the model download) |
| One ablation run, 100 steps | 63 to 83 minutes in five runs, 107 minutes in the sixth, which hit the 100-minute limit and resumed |
| Evaluation of a finished run | 10 to 16 minutes in five runs, 30 minutes in the sixth |
| Peak GPU memory in training | 6.6 GB of 15 GB |

A step takes about 68 seconds while replies are still long. It drops to 28 to 35 seconds once replies shrink, so runs where replies collapsed to about 20 tokens finish fastest. Six runs averaged 1.3 T4-hours of training and 16 minutes of evaluation each. Eight runs plus the base evaluation come to about 13 T4-hours, projected from those six. The run files are `lltCapstoneReasoner-T2-gcpT4-*` in the Run Library.

Without a GPU you can still do the analysis. Download our run files and evaluations from the Run Library and run `analyze.py` on them.

Our adapters (12 ablation runs, step 50 and final for two of them, and the two 1.7B runs) are on the Hub under `aimlcompanion/llm-training-lab/lltCapstoneReasoner/<run>/adapter-final` and `adapter-step50`. Evaluate one with `python evaluate.py --adapter hub:lltCapstoneReasoner/<run>/adapter-final`.

## Tier T3: one A100 or L4, Qwen3-1.7B-Base

Same commands with `--model Qwen/Qwen3-1.7B-Base`. Use `evaluate_vllm.py` in its own environment for the evaluations (`uv venv vllm-venv --python 3.12 && source vllm-venv/bin/activate && uv pip install vllm==0.33.0`); vLLM 0.33.0 pins its own torch (2.14.0), so it cannot share the training environment.

On one A100 40 GB, four 0.6B runs fit side by side (5.8 GB peak each at the defaults), which is how we ran the ablation. Hugging Face `generate` is bound by per-step overhead, so four side by side gave about 1.6 times the throughput of one.

## Session safety

- `train.py` writes a checkpoint every 5 steps (adapter, optimizer, grad scaler, random states). A checkpoint is written to `ckpt_new/` and then renamed, so a crash mid-write leaves the previous one intact.
- `--max_minutes` stops cleanly and saves.
- Metrics lines after the checkpoint are dropped on resume, so the curves never double count a step.

## Checklist before you start the 8 runs

- [ ] The checker tests pass and you added at least two of your own attack strings.
- [ ] The base model's pass@8 is above zero on both problem sizes. If it is near zero, GRPO has nothing to compare: fix the prompt or the difficulty first.
- [ ] A 5-step smoke run finished and resumed.
- [ ] You wrote your two predictions down.
- [ ] You worked out the GPU-hours: steps x seconds per step x 8, plus evaluation.
