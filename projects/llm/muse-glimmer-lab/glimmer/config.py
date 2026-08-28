"""Model facts and serving defaults, in one place.

Every number here comes from a primary source, and the comment says which one.
When Meta ships a point release these are the values to re-check first.

Sources:
  MODEL CARD   https://huggingface.co/meta-models/Muse-Glimmer-30B
  GGUF REPO    https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
  RESEARCH     https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model
  VLLM RECIPE  https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# -- Architecture, from the model card ----------------------------------------
# These drive the KV-cache calculator in memory.py. If you change one, the
# memory numbers change, which is the entire point of experiment 4.


@dataclass(frozen=True)
class Arch:
    """Muse Glimmer 30B, language-model side only.

    The vision tower is a separate ~1.8B ViT-G/14 and does not contribute to
    the text KV cache, so it is tracked apart from these fields.
    """

    layers: int = 52
    hidden: int = 6656
    q_heads: int = 32
    kv_heads: int = 2  # grouped-query attention, a 16:1 ratio
    head_dim: int = 128
    ffn_intermediate: int = 19_968  # SwiGLU
    vocab: int = 202_048
    rope_theta: int = 500_000  # applied to local layers only
    sliding_window: int = 2_048
    # The repeating attention pattern. Three local layers then one global.
    # 52 layers / 4 = 13 global layers, 39 local.
    attention_pattern: tuple[str, ...] = ("local", "local", "local", "global")
    default_context: int = 131_072
    max_context: int = 262_144


ARCH = Arch()

VISION_TOWER_PARAMS = 1.8e9  # ViT-G/14, 50 layers, width 1536, patch 14
MAX_VISUAL_TOKENS_PER_IMAGE = 4_096
KNOWLEDGE_CUTOFF = "2026-01-04"

# -- Sampling, from the model card and repeated by Unsloth and the vLLM recipe -
# All three sources agree, and the vLLM recipe adds the reason greedy decoding
# is a bad idea here: a reasoning model's output length varies even at a fixed
# seed, so temperature 0 buys you no reproducibility and costs you quality.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

REASONING_STRENGTHS = ("low", "medium", "high", "xhigh")

# -- Quantization footprints, from the official GGUF repo ---------------------
# Sizes are the on-disk file sizes. Runtime needs those plus the KV cache,
# which memory.py computes separately, and that gap is where most people's
# out-of-memory errors come from.
QUANTS = {
    "bf16": {
        "file_gb": 58.0,
        "bits": 16,
        "note": "Reference weights. Two 40GB A100s, or one H100 with room to spare.",
        "repo": "meta-models/Muse-Glimmer-30B",
    },
    "kquant-dynamic": {
        "file_gb": 19.7,
        "bits": 4,
        "note": "Meta's higher-quality 4-bit build, aimed at a 32GB envelope.",
        "repo": "meta-models/Muse-Glimmer-30B-GGUF",
    },
    "kquant-17gb": {
        "file_gb": 16.8,
        "bits": 4,
        "note": "Meta's 24GB build. This is the one an RTX 4090 runs.",
        "repo": "meta-models/Muse-Glimmer-30B-GGUF",
    },
    "UD-Q2_K_XL": {
        "file_gb": 13.0,
        "bits": 2,
        "note": "Unsloth dynamic 2-bit. Fits a 16GB card. Quality drops the most here.",
        "repo": "unsloth/Muse-Glimmer-30B-GGUF",
    },
    "UD-Q4_K_XL": {
        "file_gb": 17.0,
        "bits": 4,
        "note": "Unsloth dynamic 4-bit, the closest community analogue to kquant-17gb.",
        "repo": "unsloth/Muse-Glimmer-30B-GGUF",
    },
    "UD-Q6_K_XL": {
        "file_gb": 21.0,
        "bits": 6,
        "note": "Near-reference quality. RTX 5090 or a 48GB Mac.",
        "repo": "unsloth/Muse-Glimmer-30B-GGUF",
    },
}

# Companion files. Both are optional and both are commonly forgotten.
MMPROJ_FILE = "mmproj-kquant.gguf"  # 1.4 GB, required only for image input
DFLASH_FILE = "dflash-kquant.gguf"  # 1.6 GB, the speculative-decoding drafter

# -- Serving ------------------------------------------------------------------
# Any OpenAI-compatible endpoint works. llama-server, vLLM and Ollama all
# expose one, which is why this lab needs no vendor SDK.
BASE_URL = os.getenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.getenv("GLIMMER_API_KEY", "not-needed-for-local")
MODEL_NAME = os.getenv("GLIMMER_MODEL", "muse-glimmer-30B")

# Set GLIMMER_MODE=live once you have a server running. Offline is the default
# so that `uv run python scripts/01_hello.py` works on a laptop with no GPU.
MODE = os.getenv("GLIMMER_MODE", "offline").lower()


def live_mode_requested() -> bool:
    return MODE == "live"
