#!/bin/bash
# Run capstone jobs one after another on a VM whose GPU is ours alone. Each argument is one job, a shell command run
# from ~/capstoneP3; parts joined by " || " run as parallel processes. Status lines go to ~/capstoneP3/chain.status.
# Safe to re-run after a Spot preemption: train.py resumes from its checkpoint, and jobs that already wrote their
# output are skipped by the "test -f <out> ||" guards in the job lines.
cd ~/capstoneP3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export LLTRL_STEPS=~/capstoneP3/lltRL_steps
st() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> ~/capstoneP3/chain.status; }
for job in "$@"; do
  st "start $job"
  pids=()
  IFS='|' read -ra parts <<< "${job// || /|}"
  i=0
  for p in "${parts[@]}"; do
    i=$((i+1))
    bash -c "$p" > "logs/$(date -u +%H%M%S)_$i.log" 2>&1 &
    pids+=($!)
    sleep 15
  done
  ok=1
  for pid in "${pids[@]}"; do wait $pid || ok=0; done
  [ $ok = 1 ] && st "OK $job" || st "FAILED $job"
done
st "CHAIN DONE"
