#!/usr/bin/env bash
# FP8 weights, quantised on load. Real FP8 compute needs compute capability 8.9 or higher
# (L4, L40S, H100). On an A10 or T4 this runs weight-only FP8 and the speed-up is smaller.
# If your vLLM version rejects fp8_per_tensor, run `vllm serve --help=all` and use the
# spelling it lists (set FP8_FLAG).
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
exec vllm serve "$MODEL" \
  --quantization "${FP8_FLAG:-fp8_per_tensor}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
