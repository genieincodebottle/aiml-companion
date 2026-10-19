#!/usr/bin/env bash
# Cross-check: run vLLM's own benchmark at one concurrency, then your harness at the same level.
# Medians should agree within about 10 percent. If they do not, one harness is wrong, and it is
# usually yours (warm-up, which clock, how ITL was pooled).
# Flags from https://docs.vllm.ai/en/latest/cli/bench/serve/ (read 2026-10-19).
# --goodput takes KEY:VALUE pairs in milliseconds (ttft, tpot, e2el).
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
CONC="${CONC:-16}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
mkdir -p results
vllm bench serve \
  --backend vllm \
  --base-url "$BASE_URL" \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --max-concurrency "$CONC" \
  --request-rate inf \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,99 \
  --goodput ttft:2000 tpot:150 \
  --save-result --result-dir results/
echo "now run: python -m bench.cli sweep --base-url $BASE_URL --levels $CONC --name crosscheck --out results/crosscheck.json"
