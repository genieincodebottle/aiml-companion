# Runbook: Build an Instruction-Following Assistant (lltCapstoneAssistant)

Pins checked 2026-11-10: transformers 5.22.1, trl 1.17.1, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, bitsandbytes 0.50.2 for training; vLLM 0.33.0 (torch 2.14.0, transformers 5.20.2) and lm-eval 0.4.13 for generation, the judge and the IFEval checkers. Two environments, because vLLM 0.33.0 pins torch 2.14.0 and a transformers below 5.21.

## The pipeline

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1. Data | `python -m capstone.data` | Dolci-Instruct-SFT, UltraFeedback prompts, IFEval, Arena-Hard v2.0, XSTest, HelpSteer3 | `data/*.jsonl`, `data/card.json` |
| 2. SFT | `python -m capstone.sft` | `data/sft.jsonl` | `sft/adapter`, `sft/merged` |
| 3a. Samples | `python -m capstone.sample` | `sft/merged`, `data/pref_prompts.jsonl` | `pairs/samples.jsonl` |
| 3b. Labels and pairs | `python -m capstone.label` | `pairs/samples.jsonl` | `pairs/scored.jsonl`, `pairs/pairs.jsonl` |
| 4. DPO | `python -m capstone.dpo` | `sft/merged`, `pairs/pairs.jsonl` | `dpo/adapter`, `dpo/merged` |
| 5a. Evaluation replies | `python -m capstone.evalgen` | base, `sft/merged`, `dpo/merged`, Qwen3-1.7B | `eval/gen_<model>_<set>.json` |
| 5b. IFEval and XSTest scores | `python -m capstone.scores` | `eval/` | `scores/` |
| 5c. Judge | `python -m capstone.judge` | `eval/gen_*_arena.json`, `data/calib.jsonl` | `judge/` |
| 5e. Refusal judge | `python -m capstone.refusal` | `eval/gen_*_xstest.json`, XSTest human labels | `scores/refusal_labels_*.json` |
| 5d. Your labels | `python -m capstone.handlabel export`, then `score` | `eval/`, `judge/` | `handlabel/` |
| 6. Model card | `python -m capstone.card --limitations limitations.md` | `records/` | `dpo/merged/README.md` |
| 6b. Release | `python -m capstone.release prepare`, then `upload --repo <you>/<repo>` | `dpo/merged` | your Hub repository |
| Optional ablation | `ABLATION=1 bash gcp/run_chain.sh` | `data/sft.jsonl` without the Precise IF group | `sft-noif/`, `records/scores_noif.json` |

Every command takes `--tier T2|T3 --work <dir>`, and `--smoke` for a few items and steps into `<dir>-smoke`. Every stage writes `records/<stage>.json` (settings, versions, hardware, wall-clock, results). `python -m capstone.report` turns the records into Run Library files.

**Resuming.** Each stage skips outputs that exist, and training resumes from the newest checkpoint (every 50 SFT steps, every 25 DPO steps). After a timeout or a Spot preemption, run the same command again.

## T2. One free T4 (Colab or Kaggle)

Open `notebooks/lltCapstoneAssistant_T2.ipynb` with the Colab button, or upload it to Kaggle (Settings, Accelerator, GPU T4 x2, Internet on). Run the cells from the top.

- The base model is held in 4 bits (QLoRA) with fp16 compute; a T4 has no bf16.
- Smaller data: 8,000 SFT conversations of at most 1,024 tokens, 1,500 preference prompts, replies capped at 512 tokens, 250 Arena-Hard prompts.
- Reward model: Skywork-Reward-V2-Qwen3-1.7B in fp16. Judge: Olmo-3-7B-Instruct in 4 bits.
- Colab: the notebook mounts Google Drive and keeps the work there, so a new session finds it. Kaggle: use Save Version to keep `/kaggle/working`, then add that version's output as input to the next session and copy `work` back.
- To skip SFT, run `python -m capstone.fetch --what sft` (our SFT adapter, merged into the base). Say so in your report.

T2 stage times, measured on 2026-11-11 on a Google Cloud VM with one T4 in use (4 vCPUs, 26 GB of RAM, a second T4 idle), set up like Kaggle's GPU T4 x2, running the notebook's default settings. Run files: `../../runs/lltCapstoneAssistant/lltCapstoneAssistant-T2-gcpT4-notebook-*.json`.

| Stage | Time | Peak GPU memory (GiB, nvidia-smi) |
|---|---|---|
| 1. Data (CPU, downloads included) | 3 min | none |
| 2. SFT | 107 min | 4.6 |
| 3a. Samples | 189 min | 14.6, all of the T4 |
| 3b. Labels and pairs | 12 min | 4.2 |
| 4. DPO | 30 min | 4.4 |
| 5a. Evaluation replies, four models | 176 min | 10.5 |
| 5b. Scores (CPU) | under 1 min | none |
| 5c. Judge, calibration | 34 min | 14.6, all of the T4 |
| 5c. Judge, one comparison (both orders) | 129 min | 14.6, all of the T4 |
| 5e. Refusal judge | 18 min | 12.1 |

The default judge stage runs three comparisons, so about 7 h in all; we measured calibration and one comparison and projected the rest. For one comparison add `--pairs dpo:official` (about 163 min with calibration). Stage 3a writes its output only when all 1,500 prompts are done, so a stop in the middle of it starts that stage again. A first run at `GEN_BATCH=32` ran out of GPU memory on the longest prompts after 82 min; `capstone/gen.py` now splits a batch that does not fit and carries on, and the peak fills the T4, so lower `GEN_BATCH` if you share the GPU.

## T3. One cloud GPU with 40 GB or more

1. Rent a VM with one A100 40 GB (or larger) and a recent NVIDIA driver, 150 GB of disk. A Spot VM is fine: everything resumes.
2. `git clone` this repository, `cd courses/llm-training/labs/lltCapstoneAssistant`.
3. `mkdir -p ~/capstoneP2 && cp gcp/requirements-*.txt ~/capstoneP2/ && bash gcp/setup_env.sh` (builds `~/capstoneP2/venv-train` and `venv-vllm`, prints SETUP DONE).
4. `python tests/test_cpu.py` with the training venv: nine PASS lines.
5. `bash gcp/run_chain.sh --smoke`, then check every line of `logs/status` ends with `rc=0`.
6. `nohup bash gcp/run_chain.sh > chain.out 2>&1 &` and watch `logs/status`.
7. Hand labels and the model card (stage 5d and 6 above), then upload `work/dpo/merged` to your Hub account.
8. Stop the VM.

If the GPU is shared with other jobs, set `SHARED_GPU=1`: each GPU stage then waits for `/home/<user>/GPU_LOCK`, writes its own line, and removes it when the stage ends.

## T4. Full fine-tuning

Not run by us. Pass `peft_config=None` in `sft.py` and `dpo.py`, lower the learning rate by about ten times, and give `DPOTrainer` a separate frozen `ref_model`. Plan for about 16 bytes per parameter of training state in mixed precision, so 1.7B parameters need about 27 GB before activations.

## Data sources and licences (checked on the Hugging Face API 2026-11-10)

| Source | Licence | Used for |
|---|---|---|
| allenai/Dolci-Instruct-SFT | ODC-BY; each source keeps its own licence (listed on the dataset card). WildChat replies were written by GPT-4.1 | SFT |
| HuggingFaceH4/ultrafeedback_binarized | MIT | Preference prompts only |
| google/IFEval | Apache-2.0 | Evaluation |
| lmarena-ai/arena-hard-auto (arena-hard-v2.0) | Apache-2.0 | Win-rate prompts |
| XSTest, github.com/paul-rottger/xstest at commit d7bb5bd | CC-BY-4.0 | Behaviour panel |
| nvidia/HelpSteer3 (preference, validation) | CC-BY-4.0 | Judge calibration |
| Qwen/Qwen3-1.7B-Base, Qwen/Qwen3-1.7B | Apache-2.0 | Base model, comparison model |
| Skywork/Skywork-Reward-V2-Qwen3-8B and -1.7B | Apache-2.0 | Pair labels |
| openai/gpt-oss-20b, allenai/Olmo-3-7B-Instruct | Apache-2.0 | Judges |

## Safety scope

The behaviour panel uses XSTest's established prompts only. Replies to unsafe prompts are capped at 64 tokens and labelled by string match. They stay in your private work folder; `scores/xstest_<model>.json` and our public run files keep only the label. Do not publish them, and do not generate harmful text beyond the benchmark.
