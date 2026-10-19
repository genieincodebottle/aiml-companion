#!/usr/bin/env bash
# bf16 plus n-gram speculative decoding: no draft model, the proposer copies n-grams from the prompt.
# Config keys per https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ (read 2026-10-19).
# Measure acceptance from the scrape: accepted draft tokens divided by proposed draft tokens
# (the vllm:spec_decode_* counters). Check which names your version exposes first.
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
K="${NUM_SPEC_TOKENS:-4}"
LOOKUP="${PROMPT_LOOKUP_MAX:-4}"
exec vllm serve "$MODEL" \
  --dtype "${DTYPE:-bfloat16}" \
  --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${K},\"prompt_lookup_max\":${LOOKUP}}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
