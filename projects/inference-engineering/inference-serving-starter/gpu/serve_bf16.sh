#!/usr/bin/env bash
# Baseline: bf16 weights. On a T4 (no bf16) set DTYPE=half.
# Needs: MODEL (default below). Gated models also need HF_TOKEN in the environment.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
exec vllm serve "$MODEL" \
  --dtype "$DTYPE" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
