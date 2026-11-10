#!/bin/bash
# The lltCapstoneAssistant reference run on one cloud GPU (T3). Resumable: every stage skips work it already finished,
# so after a Spot preemption you start the VM and run the same command again.
#   ./run_chain.sh --smoke     a few items and steps per stage, to test the chain (about 15 GPU-min, est.)
#   ./run_chain.sh             the full reference run
# On a shared GPU, set SHARED_GPU=1: each GPU stage then waits for /home/rajsr/GPU_LOCK to be free, writes its own lock
# line (owner + expected end), and removes it when the stage ends. On a GPU of your own, leave it unset.
set -u
cd "$(dirname "$0")/.."
T=$HOME/capstoneP2/venv-train/bin/python
V=$HOME/capstoneP2/venv-vllm/bin/python
ARGS="--tier T3 --work $HOME/capstoneP2/work $*"
LOCK=/home/rajsr/GPU_LOCK
ME=capstoneP2
mkdir -p logs
# flashinfer (inside vLLM) compiles kernels on first use and needs ninja on PATH
export PATH=$HOME/capstoneP2/venv-vllm/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

gpu_busy() { [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 2000 ]; }
acquire() {
  while true; do
    if [ -f $LOCK ] && ! grep -q "^$ME" $LOCK; then echo "$(date -u +%T) waiting, lock: $(cat $LOCK)"; sleep 30; continue; fi
    if gpu_busy && ! grep -q "^$ME" $LOCK 2>/dev/null; then echo "$(date -u +%T) waiting, GPU memory in use"; sleep 30; continue; fi
    echo "$ME $1 until $(date -u -d "+$2 min" +%Y-%m-%dT%H:%M:%SZ)" > $LOCK
    sleep 3
    grep -q "^$ME" $LOCK && return 0
  done
}
release() { grep -q "^$ME" $LOCK 2>/dev/null && rm -f $LOCK; return 0; }
trap release EXIT
trap "exit 130" INT TERM

run() {  # run <name> <est minutes, 0 = CPU only> <python> <module> [extra args]
  name=$1; est=$2; py=$3; mod=$4; shift 4
  [ "$est" != 0 ] && [ -n "${SHARED_GPU:-}" ] && acquire "$name" "$est"
  echo "$(date -u +%FT%TZ) start $name" >> logs/status
  $py -m capstone.$mod $ARGS "$@" > logs/$name.log 2>&1
  rc=$?
  [ "$est" != 0 ] && [ -n "${SHARED_GPU:-}" ] && release
  echo "$(date -u +%FT%TZ) end $name rc=$rc" >> logs/status
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> logs/status 2>/dev/null
  [ $rc -eq 0 ] || { echo "stage $name failed, see logs/$name.log"; tail -20 logs/$name.log; exit $rc; }
}

run data      0   $T data
run sft       60  $T sft
run sample    30  $V sample
run label     40  $T label --second Skywork/Skywork-Reward-V2-Qwen3-1.7B
run dpo       40  $T dpo
run evalgen   45  $V evalgen
run scores    0   $V scores
run judge     90  $V judge --judges openai/gpt-oss-20b,allenai/Olmo-3-7B-Instruct
run refusal   20  $V refusal
echo "$(date -u +%FT%TZ) CHAIN DONE" >> logs/status

# Optional ablation (our reference run did it): SFT without the Precise IF group, then IFEval only.
if [ -n "${ABLATION:-}" ]; then
  run sft-noif      90  $T sft --name sft-noif --drop-group precise_if
  run evalgen-noif  15  $V evalgen --models sft-noif --sets ifeval --record evalgen_noif
  run scores-noif   0   $V scores --models base,sft,sft-noif --record scores_noif
  echo "$(date -u +%FT%TZ) ABLATION DONE" >> logs/status
fi
