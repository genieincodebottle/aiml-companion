# Runbook for the H100 reference run (lltCapstoneGPT2)

For the owner. This runbook trains our 124M GPT on 10B FineWeb-Edu tokens on one rented H100, on your own account. It produces the checkpoints the learner page needs, plus the Run Library files. Prices were read on 2026-11-05. Recheck them on the provider's page before you start.

## What it produces

- The main run, 19,073 steps x 524,288 tokens = 10.0B tokens. It uses a WSD schedule, which means warmup over 700 steps, a constant LR of 6e-4 until step 17,166, then a linear decay to 0.
- Full training-state checkpoints at the stable-phase branch points: steps 906, 1,813, 4,292, 4,529, 9,060 and 17,166, plus the final step 19,073.
  - **Step 4,529 is the checkpoint the free learner path loads.** The learner runs the last 240 steps of a 2.5B-token budget on Kaggle.
- bf16 weights every 1,000 steps. These give the generation timeline and the HellaSwag curve.
- Five branch decays, one model per token budget: 0.5B, 1B, 2.5B with 5% decay, 2.5B with 10% decay, and 5B.
- The lm-eval 0.4.13 scores of every model and of `openai-community/gpt2` on the same 4 tasks (HellaSwag, ARC-Easy, PIQA, LAMBADA).
- `deliverables.tar.gz`, which holds the Run Library JSON, logs, evals and sizes.
- The learner files on the Hub, in `aimlcompanion/llm-training-lab/lltCapstoneGPT2/` (step 7).

The scripts are in this folder: `run_reference.sh`, `prep_fineweb.py`, `train.py`, `gpt.py`, `eval.py`, `hf_wrapper.py`, `calib_summary.py`, `hf_upload_capstone.py`. The resume logic in `train.py` was tested on Kaggle 2x T4 on 2026-11-05 (run `lltCapstoneGPT2-T2-resume-check`). Two changes came after that test and were checked only on CPU: checkpoints now load with `weights_only=True`, and `--resume` also accepts a downloaded `state.pt` file.

## 1. Rent the pod

| Choice | Value |
|---|---|
| GPU | 1x H100 80 GB **SXM**. PCIe also works but is slower; its throughput is unmeasured |
| Image | Any image with PyTorch 2.4 or newer and CUDA 12+. Examples: RunPod's PyTorch template, or Lambda's default image (Lambda Stack) |
| Disk | **150 GB** at `/workspace`. Peak use is about 50 GB during data prep, then about 20 GB of tokens plus about 30 GB of checkpoints and weights (est.) |
| vCPUs and RAM | Whatever the H100 pod comes with. Data prep uses half the vCPUs |

On-demand prices, 2026-11-05:

| Provider | H100 SXM | H100 PCIe | Source |
|---|---|---|---|
| RunPod Community | from $2.69/h | from $1.99/h | https://www.runpod.io/pricing |
| RunPod Secure | $3.99/h | $2.89/h | same |
| Lambda | $4.29/h | $3.29/h | https://lambda.com/pricing (the page says these prices take effect 2026-11-16) |

## 2. Copy the scripts onto the pod

From your machine, send the folder `courses/llm-training/labs/lltCapstoneGPT2` to `/workspace/lltCapstoneGPT2`. Use the provider's SSH (`scp -r`), or `runpodctl send` on RunPod. Then run:

```
cd /workspace/lltCapstoneGPT2
read -rs HF_TOKEN && export HF_TOKEN   # optional READ token, avoids download rate limits; not saved anywhere
```

## 3. Calibrate first (about 20 min, about $0.90-1.35 on RunPod H100 SXM)

```
PRICE_PER_HOUR=2.69 bash run_reference.sh calibrate
```

Set `PRICE_PER_HOUR` to the rate you pay. Calibration does four things:
- Tokenizes 1 of the 14 files and records how long that takes.
- Trains 40 steps at micro-batch 32, then at 64.
- Scores `gpt2` in lm-eval.
- Prints `calibrate_summary.txt`, which gives the faster micro-batch, tokens/s, MFU, peak memory, and the projected hours and dollars for the full run.

Continue only if the projection is within the budget you approved. Our estimate before calibration is 6-8 h of training at 30-40% MFU, the plan's planning band (section 2.2 of DESIGN.md).

## 4. Start the full run (one command)

```
cd /workspace/lltCapstoneGPT2
MICRO=<value from calibration, 32 or 64> nohup bash run_reference.sh full > /workspace/run/main.log 2>&1 &
```

`nohup` keeps the run alive if your SSH session drops.

The script runs these stages in order:
1. Data prep: 14 files, about 10B tokens.
2. Training.
3. The five branch decays.
4. Evals.
5. Packaging.

**If anything stops, run the same command again.** Prep is skipped once `index.json` exists. Training resumes from the newest checkpoint, which is saved every 500 steps and at each branch point. Finished branches and evals are skipped.

## 5. Watch it

```
tail -f /workspace/run/main.log                         # every 10 steps: loss, lr, tokens/s, MFU, memory
grep valLoss /workspace/run/train.log | tail -5         # validation loss every 250 steps
nvidia-smi                                              # GPU busy and memory
ls /workspace/run/ckpt /workspace/run/weights           # checkpoints so far
```

What you should see:
- Train loss starts near 11.0. ln 50,257 is 10.82, and our measured first step was 11.01. It falls slowly during the 700 warmup steps.
- tokens/s matches the calibration figure.
- The `lossScale` field is absent, because bf16 needs no loss scaling.

If tokens/s drops well below calibration, check `nvidia-smi` for another process on the GPU.

## 6. Time and cost (est., replaced by your calibration)

| Stage | Time | RunPod Community $2.69/h | RunPod Secure $3.99/h |
|---|---|---|---|
| Calibration | 20 min | $0.90 | $1.33 |
| Data prep | under 1 h (measured on 1 file in calibration) | up to $2.69 | up to $3.99 |
| Main training, 10.0B tokens | 6.0-8.0 h | $16.1-21.5 | $23.9-31.9 |
| Branch decays, 1,337 steps, 0.70B tokens | 0.4-0.6 h | $1.1-1.5 | $1.7-2.2 |
| Evals: final and gpt2, 5 budget models, HellaSwag curve over about 26 checkpoints | under 1 h | up to $2.69 | up to $3.99 |
| **Total** | **about 7-11 h** | **about $18-30** | **about $27-44** |

Stop the pod as soon as step 8 is done. Billing runs while the pod exists, even when it is idle.

## 7. Upload the learner files to the Hub (from the pod)

Target: the public model repo `aimlcompanion/llm-training-lab`, folder `lltCapstoneGPT2/`.

1. On huggingface.co, create a **fine-grained token** with write access to `aimlcompanion/llm-training-lab` only.
2. On the pod, give the token to this shell only. It is never written to a file, the repo or the shell history:

```
read -rs HF_TOKEN && export HF_TOKEN       # paste the token, press Enter (nothing is echoed)
cd /workspace/lltCapstoneGPT2
python hf_upload_capstone.py --dry-run     # lists every file, its size and sha256; uploads nothing
python hf_upload_capstone.py               # one commit to aimlcompanion/llm-training-lab/lltCapstoneGPT2/
unset HF_TOKEN
```

`hf auth login` also works, but it caches the token in a file on the pod (`~/.cache/huggingface/token`). If you use it, run `hf auth logout` right after the upload. (`huggingface-cli` no longer exists in huggingface_hub 1.x and 2.x; the command is `hf`.) Revoke the token on huggingface.co once the upload is done.

What goes up (sizes from the measured checkpoint format; weights are bf16):

| Repo file | Source on the pod | Size | Used by |
|---|---|---|---|
| `step_004529_state.pt` | `run/ckpt/step_004529/state.pt` | 1.483 GB | Free path: full training state at the end of the stable phase |
| `weights_10B_final.pt` | `run/weights/weights_step_019073.pt` | about 0.25 GB | The final model |
| `weights_0p5B.pt`, `weights_1B.pt`, `weights_2p5B.pt`, `weights_2p5B_d10.pt`, `weights_5B.pt` | `branches/*/weights/` | about 0.25 GB each | Budget comparison; `weights_2p5B.pt` is the free path's target |
| `runs/*.json`, `runs/evals/*.json` | Run Library files and evals | small | Run Explorer |
| `manifest.json` | written by the script | small | Size and sha256 of every file |

Every `.pt` file holds only tensors and plain Python types, so notebooks load it with `torch.load(path, map_location="cpu", weights_only=True)`. The RNG files stay on the pod; a learner run on a different GPU layout does not use them.

The free Kaggle path downloads from `https://huggingface.co/aimlcompanion/llm-training-lab/resolve/main/lltCapstoneGPT2/<file>`, checks the sha256 against `manifest.json`, then runs:

```
torchrun --nproc_per_node=2 train.py --data data --out decay --micro 8   --resume step_004529_state.pt --stable-end 4529 --max-steps 4769 --reset-loader-shard 1
```

## 8. Download the rest and send it back

```
scp <pod>:/workspace/deliverables.tar.gz .                     # small: JSON, logs, evals
scp -r <pod>:/workspace/run/ckpt <pod>:/workspace/branches .   # optional: every other checkpoint (about 10 GB, est.)
```

On RunPod you can also use `runpodctl send <path>` on the pod and `runpodctl receive <code>` locally.

Send us:
1. `deliverables.tar.gz`.
2. The commit URL that `hf_upload_capstone.py` printed.
3. The hourly price you paid and the total billed amount, for the Run Library `costUSD` and `priceCheckedOn` fields.
4. The provider and the exact GPU name (in `nvidia-smi.txt` inside the archive).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `need torch >= 2.4` | Old image | Pick a newer PyTorch image; do not pip-install torch over the image's CUDA build |
| `pip` refuses with "externally-managed-environment" | System Python | `python -m venv /workspace/venv --system-site-packages && source /workspace/venv/bin/activate`, then rerun |
| CUDA out of memory at micro 64 | Batch too big | `MICRO=32` (calibration tells you which fits) |
| Hugging Face 429 or slow download | Anonymous rate limit | `read -rs HF_TOKEN && export HF_TOKEN` with a read token, then rerun; prep restarts the unfinished files |
| Pod restarted mid-run | Host maintenance | Rerun step 4's command; it resumes from the newest checkpoint |
| Disk full | Volume under 150 GB | Grow the volume; or delete `/workspace/data_cal` and `/workspace/cal*` from calibration |
