# Runbook on Google Cloud: one H100 with Flex-start (lltCapstoneGPT2)

For the owner. This is the Google Cloud version of `RUNBOOK.md`. Training, monitoring, the Hub upload and what to send back are the same; this file covers what differs on Google Cloud. Facts were checked on Google's documentation on 2026-11-06. Recheck the price on the pricing page before you start.

## What you rent

| Choice | Value | Why |
|---|---|---|
| Machine | `a3-highgpu-1g`, one H100 80 GB SXM | Same GPU as the runbook's estimate |
| Provisioning | **Flex-start** | Google sells A3 machines with fewer than 8 GPUs only as Spot or Flex-start. Flex-start runs without interruption once it starts; Spot can be stopped at any time |
| Price | About **$4.79 per hour** (Dynamic Workload Scheduler pricing page; Google says Flex-start prices change in November 2026, so check it) | Whole run about 7.5-11.5 h, so about $36-55, plus about $1-2 for disk and the upload |
| Image | Deep Learning VM `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (project `deeplearning-platform-release`) | PyTorch 2.9, CUDA 12.9, NVIDIA driver preinstalled. The scripts need torch 2.4 or newer |
| Boot disk | 300 GB | The image uses part of it; the run needs about 150 GB at peak |
| Zone | `us-central1-a`, `-b` or `-c` | Zones Google lists for A3 High in us-central1 |

## Step 0. One-time account setup (all in the Cloud console, about 15 min plus quota wait)

1. **Billing.** The account is a paid billing account with Google for Startups credit (owner, 2026-11-06), so no upgrade is needed. In Billing, Credits, check that the credit applies to Compute Engine and note its expiry date. (A free-trial account would first need an upgrade: free-trial accounts cannot use GPUs.)
2. **Project.** Create a project for this run only, for example `llt-capstone`, so the run is easy to find and delete. Do not use the AWS production account or any project that hosts the site.
3. **Enable the Compute Engine API** for that project.
4. **Budget alert.** Billing, Budgets and alerts: a budget of $60 on this project with alerts at 50%, 90% and 100%.
5. **Quota.** IAM and Admin, Quotas. Filter by "H100" and "preemptible" for your region (for example us-central1). Request:
   - Preemptible NVIDIA H100 GPUs in the region: **1**
   - GPUs (all regions): **1**
   - Preemptible CPUs in the region: enough for the machine's vCPUs (check `a3-highgpu-1g` on the machine-types page; ask for 32)

   Flex-start uses preemptible quota. Approval can take from hours to a few days. Without it, the create request stays pending and then fails.

## Step 1. Open a shell with gcloud

Use **Cloud Shell** (the terminal icon at the top of the console): it has `gcloud` installed and logged in. Set the project:

```
gcloud config set project llt-capstone
```

## Step 2. Request the VM (Flex-start)

```
gcloud compute instances create llt-capstone-h100 \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-1g \
    --provisioning-model=FLEX_START \
    --request-valid-for-duration=2h \
    --max-run-duration=20h \
    --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=300GB
```

- The request waits up to 2 hours for a free H100 (`PENDING`). Check it with `gcloud compute instances list`. When capacity appears, the VM starts within about two minutes. Billing is pay as you go.
- If the request expires, run it again, or try `--zone=us-central1-b` or `us-central1-c`.
- `--max-run-duration=20h` is a safety cap above the expected 7.5-11.5 h. At the cap the VM stops and keeps its disk; the training resumes from its last checkpoint when you start it again.
- Google has not published a combined example for a3-highgpu-1g with Flex-start; the flags come from its Flex-start and A3 pages. If gcloud rejects a flag, read its error and tell Claude.

**Fallback, Spot** (cheaper per hour, but Google can stop it at any time):

```
gcloud compute instances create llt-capstone-h100 \
    --zone=us-central1-a --machine-type=a3-highgpu-1g \
    --provisioning-model=SPOT --instance-termination-action=STOP \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release --boot-disk-size=300GB
```

If a Spot VM is stopped, start it again (`gcloud compute instances start llt-capstone-h100 --zone=us-central1-a`), SSH in and rerun the step 4 command: the run resumes from the newest checkpoint (one is saved every 500 steps, about every 10 minutes on an H100).

## Step 3. Log in and get the scripts

```
gcloud compute ssh llt-capstone-h100 --zone=us-central1-a
```

On the VM:

```
nvidia-smi                                    # one H100 80GB
sudo mkdir -p /workspace && sudo chown "$USER" /workspace
cd /workspace
git clone --depth 1 https://github.com/genieincodebottle/aiml-companion.git
cp -r aiml-companion/courses/llm-training/labs/lltCapstoneGPT2 /workspace/lltCapstoneGPT2
cd /workspace/lltCapstoneGPT2
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

If `python` is not the image's PyTorch environment, run `conda activate base` (or the environment the image login message names) first. If `pip` refuses with "externally-managed-environment", see the troubleshooting table in `RUNBOOK.md`.

Optional, avoids Hugging Face download rate limits during data prep (a READ token, kept only in this shell):

```
read -rs HF_TOKEN && export HF_TOKEN
```

## Step 4. Calibrate, then run (same as RUNBOOK.md sections 3 and 4)

```
PRICE_PER_HOUR=4.79 bash run_reference.sh calibrate          # about 20 min, about $1.60
cat /workspace/run/calibrate_summary.txt                      # micro-batch, tokens/s, projected hours and dollars
```

Continue only if the projection fits your budget. Then:

```
MICRO=<value from calibration> nohup bash run_reference.sh full > /workspace/run/main.log 2>&1 &
```

You can close the SSH window; the run keeps going. To watch it later, SSH in again and use `tail -f /workspace/run/main.log` (RUNBOOK.md section 5 says what to expect). If anything stops, run the same command again: it resumes.

## Step 5. Upload the learner files to the Hub

Exactly RUNBOOK.md section 7, on the VM (`hf_upload_capstone.py --dry-run`, then the real upload, with a fine-grained WRITE token for `aimlcompanion/llm-training-lab` only, then revoke the token).

## Step 6. Bring the results home and delete the VM

From Cloud Shell:

```
gcloud compute scp llt-capstone-h100:/workspace/deliverables.tar.gz . --zone=us-central1-a
```

Download `deliverables.tar.gz` from Cloud Shell to your laptop (Cloud Shell menu, Download file). Then **delete the VM, which also deletes its boot disk** (billing stops only when it is gone):

```
gcloud compute instances delete llt-capstone-h100 --zone=us-central1-a
gcloud compute instances list                                 # must show nothing
```

Check Billing a day later for the final amount.

## Step 7. Send to Claude

1. `deliverables.tar.gz`.
2. The commit URL that `hf_upload_capstone.py` printed.
3. The hourly price and the total billed amount, and that it ran on Google Cloud `a3-highgpu-1g` Flex-start in zone X.

Claude then builds the capstone page from these numbers (the full rented-H100 path and the free Kaggle path that starts from the step 4,529 checkpoint).

## Troubleshooting (Google Cloud specific)

| Symptom | Cause | Fix |
|---|---|---|
| Request stays `PENDING`, then fails | No free H100 in that zone within 2 h, or no quota | Check the quota page; retry later or in another zone (`-b`, `-c`) |
| `Quota 'PREEMPTIBLE_NVIDIA_H100_GPUS' exceeded` (or similar) | Quota not granted yet | Request it on the Quotas page (step 0) |
| Error says GPUs are not available for the account | Free-trial billing | Upgrade to a paid billing account (step 0) |
| VM stopped by itself | Hit `--max-run-duration`, or a Spot preemption | `gcloud compute instances start ...`, SSH in, rerun the step 4 command |
| Disk full | Boot disk too small | Delete `/workspace/data_cal` and `/workspace/cal*`, or resize the disk in the console |
