#!/usr/bin/env bash
# 4-bit AWQ. Use a checkpoint that is ALREADY quantised; vLLM reads the method from its config.
# The default below is a community AWQ build of Llama 3.1 8B Instruct: check that the repo exists
# and read its licence before you rent a GPU. Any AWQ checkpoint of the same family works.
set -euo pipefail
MODEL_AWQ="${MODEL_AWQ:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
exec vllm serve "$MODEL_AWQ" \
  --dtype "${DTYPE:-half}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
