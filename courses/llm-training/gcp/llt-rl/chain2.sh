#!/bin/bash
# chain2: like chain.sh, but keeps ~/GPU_LOCK while it runs and removes it only when no other lltRL chain runs.
# Run lltRL jobs one after another under ~/GPU_LOCK. Each argument is one job line, e.g. "smoke out_smoke" or "grpo out_x model=... lr=...".
# Several words separated by " || " in one argument run as parallel processes of the same job. Status in ~/llt-rl/chain.status.
cd ~/llt-rl
source ~/llt-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> chain.status; }
while [ -f ~/GPU_LOCK ] && ! grep -q lltRL ~/GPU_LOCK; do st "waiting for lock: $(cat ~/GPU_LOCK)"; sleep 120; done
echo "lltRL $(date -u +%Y-%m-%dT%H:%M:%SZ)" > ~/GPU_LOCK
( while true; do sleep 60; [ -f ~/GPU_LOCK ] || echo "lltRL $(date -u +%Y-%m-%dT%H:%M:%SZ)" > ~/GPU_LOCK; done ) &
keeper=$!
for job in "$@"; do
  st "start $job"
  pids=()
  IFS='|' read -ra parts <<< "${job// || /|}"
  for p in "${parts[@]}"; do
    set -- $p
    mkdir -p "$2"
    python run_job.py $p > "$2/run.log" 2>&1 &
    pids+=($!)
  done
  ok=1
  for pid in "${pids[@]}"; do wait $pid || ok=0; done
  [ $ok = 1 ] && st "OK $job" || st "FAILED $job"
done
kill $keeper
pgrep -f run_job.py > /dev/null || rm -f ~/GPU_LOCK      # another lltRL chain still running keeps the lock
st "CHAIN DONE"
