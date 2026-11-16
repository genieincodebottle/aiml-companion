#!/usr/bin/env bash
# Tensor parallel across 2 GPUs of ONE machine: the same bf16 baseline, split over two cards.
# Needs 2 visible GPUs. Check how they are linked first: `nvidia-smi topo -m` shows NV# (NVLink)
# or PHB/PIX/SYS (PCIe). Over PCIe every layer pays an all-reduce, so TP=2 often gives well under
# 2x the tokens per second; measure it, do not assume it.
# Flags: --tensor-parallel-size per https://docs.vllm.ai/en/latest/serving/parallelism_scaling/
# (read 2026-11-16). That page says to use tensor parallelism when a model is too big for one
# GPU but fits on one node, and suggests pipeline parallelism for cards without NVLink such as
# the L40S. An 8B model fits on one 24 GB card, so here TP=2 is a measurement exercise, not a need.
# Run it next to serve_bf16.sh with the same flags and compare the two sweeps. Set NCCL_DEBUG=INFO
# to see which transport NCCL chose (the docs use NCCL_DEBUG=TRACE for the same check).
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
exec vllm serve "$MODEL" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
  --enable-prefix-caching \
  --port "${PORT:-8000}"
