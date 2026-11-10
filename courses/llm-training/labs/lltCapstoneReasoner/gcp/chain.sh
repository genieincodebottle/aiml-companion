#!/bin/bash
# Run capstone jobs one after another while holding ~/GPU_LOCK (the GPU is shared with other labs).
# Each argument is one job: a shell command run from ~/capstoneP3. Parts joined by " || " run as parallel
# processes of the same job. EXPECT_MIN (minutes) goes into the lock so other users know when it frees.
# Re-running the same chain after a preemption is safe: train.py continues from its last checkpoint and
# evaluate.py jobs are skipped when their output exists (wrap them in skip_if_exists below).
cd ~/capstoneP3
source ~/llt-venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export LLTRL_STEPS=~/capstoneP3/lltRL_steps
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> ~/capstoneP3/chain.status; }
# take the lock atomically (noclobber), so two waiting chains cannot both write it
while true; do
  END=$(date -u -d "+${EXPECT_MIN:-60} minutes" +%Y-%m-%dT%H:%MZ)
  if grep -qs lltCapstoneReasoner ~/GPU_LOCK; then echo "lltCapstoneReasoner start $(date -u +%Y-%m-%dT%H:%M:%SZ) expected end $END" > ~/GPU_LOCK; break; fi
  if ( set -o noclobber; echo "lltCapstoneReasoner start $(date -u +%Y-%m-%dT%H:%M:%SZ) expected end $END" > ~/GPU_LOCK ) 2>/dev/null; then break; fi
  st "waiting for lock: $(cat ~/GPU_LOCK 2>/dev/null)"; sleep 90
done
st "lock taken, expected end $END"
for job in "$@"; do
  st "start $job"
  pids=()
  IFS='|' read -ra parts <<< "${job// || /|}"
  i=0
  for p in "${parts[@]}"; do
    i=$((i+1))
    bash -c "$p" > "logs/$(date -u +%H%M%S)_$i.log" 2>&1 &
    pids+=($!)
    sleep 20     # stagger model loads
  done
  ok=1
  for pid in "${pids[@]}"; do wait $pid || ok=0; done
  [ $ok = 1 ] && st "OK $job" || st "FAILED $job"
done
rm -f ~/GPU_LOCK
st "CHAIN DONE"
