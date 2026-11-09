# lltDistillation notebook check on the Google Cloud T4 VM (llt-c2)

Runs the lab notebook with its learner defaults, plus `USE_OUR_ADAPTER = True` in step 7, in one Python process. It is the
only part of the page code that has not executed on a GPU yet:

- step 4 with `USE_OUR_TEACHER_DATA = True` (the default): downloads `lltDistillation/teacher_train.json` from the Hub and
  checks its 2,048 questions against the seed-0 shuffle;
- step 7 with `USE_OUR_ADAPTER = True`: downloads `lltDistillation/onpolicy-reverse-s0/`, loads it on the fp16 student,
  writes it into the plain student with `sync` and evaluates it. Expected accuracy 0.532 on the 500 test questions (the
  Run Library value; fp16 kernels on another torch build may move a few answers).

Every other cell already ran end to end on a Kaggle T4 in `llt-distill-main` v1 (2026-11-08). `RUN_EXPERIMENT` stays False.
`llt-distill-notebook.ipynb` is built by `labs/lltDistillation/build_notebook.py` (variant `default`) from the page code in
`labs/lltDistillation/steps/`. Do not edit it; edit the steps and rebuild.

## Prerequisite

Both Hub files must exist (public, no token needed):

```bash
curl -s "https://huggingface.co/api/models/aimlcompanion/llm-training-lab/tree/main/lltDistillation?recursive=true"
```

It must list `lltDistillation/teacher_train.json` and `lltDistillation/onpolicy-reverse-s0/adapter_model.safetensors`.

## Commands (on llt-c2, project llt-capstone)

```bash
# copy the folder up (from the laptop, in C:/projects/aiml-companion)
gcloud compute scp --project llt-capstone --zone us-central1-a --recurse courses/llm-training/gcp/llt-distill-notebook llt-c2:~/

# on the VM
cd ~/llt-distill-notebook
python3 --version                                         # must be 3.11 or newer; run_nb.py asserts it
python3 -m venv --system-site-packages ~/llt-venv        # keeps the image's CUDA torch (reuse the venv if the DPO job made it)
source ~/llt-venv/bin/activate
pip uninstall -y -q torchao 2>/dev/null; pip install -q -r requirements.txt
python -c "import torch, transformers, trl, peft, bitsandbytes; print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, trl.__version__, peft.__version__, bitsandbytes.__version__)"
nohup python run_nb.py --out outputs > job.log 2>&1 &
tail -f job.log                                           # "CELL cellNN_stepK start/end <s>" lines; "JOB DONE" at the end
```

Spot preemption: start the VM and run the same `nohup python run_nb.py ...` line again. The job restarts from the
beginning (about 50 GPU-min in total, no single step longer than 17 min), so no checkpoint state is kept.

## Outputs to copy back

```bash
gcloud compute scp --project llt-capstone --zone us-central1-a llt-c2:~/llt-distill-notebook/outputs/distill_results.json llt-c2:~/llt-distill-notebook/outputs/notebook_stdout.txt llt-c2:~/llt-distill-notebook/job.log ./courses/llm-training/kaggle/out/gcp-llt-distill-notebook/
```

The `outputs/adapters/` folder (two 39 MB adapters from steps 5 and 6) need not be copied.

## Then, on the laptop

```bash
cd courses/llm-training/labs/lltDistillation
../../.venv-latest/Scripts/python.exe record_gpu.py ../../kaggle/out/gcp-llt-distill-notebook notebook "GCP T4 llt-c2, run_nb.py, learner defaults + USE_OUR_ADAPTER = True"
```

That writes `runs/lltDistillation/lltDistillation-T2-notebook.json`. Ledger lines (`llt5/kaggle_ledger.csv`):
`start|end,<ISO time>,lltDistillation,gcp:llt-distill-notebook,1,<min>`.

## Acceptance

- `job.log` ends with `JOB DONE`, no traceback.
- Step 4 prints `teacher answers 2048 | correct 0.817 | ended at the stop string 0.893 | mean tokens 96.4` (the download
  path recomputes these from the file, so they must match exactly).
- Step 7 prints the on-policy arm's accuracy near 0.532 (Run Library value; within about 1 point).
- `KERNEL peak GB over the whole session` below 15.

## Expected GPU time (from the Kaggle T4 run llt-distill-main v1, per-cell end times)

| Part | Basis | Minutes |
|---|---|---|
| Install and model downloads (Qwen3-4B-Base about 8 GB, Qwen3-0.6B-Base 1.2 GB, GSM8K) | not timed on Kaggle | 5 to 8 |
| Steps 1 to 3: NumPy fit, TRL kernel check on CPU, load both models, evaluate teacher and student on 500 questions | cells 1 to 4: 1,000 s | 17 |
| Step 4, download the teacher's answers | 1.5 MB | under 1 |
| Step 5, SFT on the teacher's answers, 64 steps, then evaluate | cell 6: 649 s | 11 |
| Step 6, logit KD with forward KL, 64 steps, then evaluate | cell 7: 974 s | 16 |
| Step 7, load our adapter and evaluate; TRL DistillationTrainer, 3 steps | eval about 230 s; cell 9: 80 s | 5 |
| Steps 8 and 9 (RUN_EXPERIMENT False: tables only) | | under 1 |
| Total | | about 55 to 60 |

The VM image's torch differs from Kaggle's (2.11.0+cu130). The record keeps the versions the job printed; the page quotes
Kaggle times for the lab card and uses this run only to confirm the two download paths and their numbers.
