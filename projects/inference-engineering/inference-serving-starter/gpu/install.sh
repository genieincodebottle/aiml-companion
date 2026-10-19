#!/usr/bin/env bash
# Pinned vLLM install. Docs read 2026-10-19. v0.31.0 was released 2026-10-05.
set -euo pipefail
VLLM_VERSION=0.31.0
python3 -m pip install --upgrade uv
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto
vllm download-kernels     # FlashInfer precompiled kernels, avoids a long compile at first start
vllm --version
