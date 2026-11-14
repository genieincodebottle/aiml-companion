#!/usr/bin/env bash
# Pinned vLLM install. v0.33.0 was released 2026-11-02; this script and the serve_*.sh /
# bench_crosscheck.sh scripts were run with it on a T4 on 2026-11-14 (see gpu/README.md).
set -euo pipefail
VLLM_VERSION=0.33.0
python3 -m pip install --upgrade uv
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
vllm download-kernels     # FlashInfer precompiled kernels, avoids a long compile at first start
vllm --version
