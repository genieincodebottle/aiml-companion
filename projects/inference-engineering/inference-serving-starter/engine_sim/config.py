"""Hardware, model and engine configuration for engine_sim.

Every number here is either a vendor figure (with the page it came from, read
2026-10-19) or a clearly labelled modelling assumption. Nothing is measured on a
real GPU. Compute figures are DENSE (vendor "with sparsity" figures are halved).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GPUSpec:
    key: str
    name: str
    mem_gb: float          # decimal GB, as on the vendor page
    hbm_gbps: float        # peak memory bandwidth, GB/s
    bf16_tflops: float     # dense BF16/FP16 tensor-core peak
    fp8_tflops: float      # dense FP8 tensor-core peak, 0 = no FP8 tensor cores
    source: str


# NVIDIA L4: 24 GB, 300 GB/s, BF16 242 and FP8 485 TFLOPS "shown with sparsity;
# one-half lower without" -> 121 and 242 dense.
#   https://www.nvidia.com/en-us/data-center/l4/
# NVIDIA A10: 24 GB GDDR6, 600 GB/s, BF16 250 TFLOPS with sparsity (125 without).
# The A10 page lists no FP8 support (Ampere has no FP8 tensor cores).
#   https://www.nvidia.com/en-us/data-center/products/a10-gpu/
# NVIDIA H100 SXM: 80 GB HBM3, 3.35 TB/s, BF16 989 and FP8 1979 TFLOPS dense.
#   https://www.nvidia.com/en-us/data-center/h100/
GPUS: dict[str, GPUSpec] = {
    "l4": GPUSpec("l4", "NVIDIA L4", 24, 300, 121, 242, "https://www.nvidia.com/en-us/data-center/l4/"),
    "a10": GPUSpec("a10", "NVIDIA A10", 24, 600, 125, 0, "https://www.nvidia.com/en-us/data-center/products/a10-gpu/"),
    "h100": GPUSpec("h100", "NVIDIA H100 SXM", 80, 3350, 989, 1979, "https://www.nvidia.com/en-us/data-center/h100/"),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    params: float          # parameter count
    layers: int
    kv_heads: int
    head_dim: int
    source: str

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        # one key and one value, per layer, per KV head (grouped-query attention)
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


# Llama 3.1 8B: 8.03B params, 32 layers, 8 KV heads, head_dim 128 (model card + config.json).
# Llama 3.1 70B: 70.6B params, 80 layers, 8 KV heads, head_dim 128.
#   https://huggingface.co/meta-llama/Llama-3.1-8B
#   https://huggingface.co/meta-llama/Llama-3.1-70B
MODELS: dict[str, ModelSpec] = {
    "8b": ModelSpec("8b", "Llama-3.1-8B", 8.03e9, 32, 8, 128, "https://huggingface.co/meta-llama/Llama-3.1-8B"),
    "70b": ModelSpec("70b", "Llama-3.1-70B", 70.6e9, 80, 8, 128, "https://huggingface.co/meta-llama/Llama-3.1-70B"),
}

# Bytes per weight parameter. int4 is 4-bit weights plus group scales and zeros
# (modelling assumption: about 0.03 byte/param at group size 128).
QUANT_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0, "int4": 0.53}

BLOCK_SIZE = 16  # tokens per KV block, the vLLM default


@dataclass(frozen=True)
class EngineConfig:
    model: str = "8b"
    gpu: str = "l4"
    num_gpus: int = 1
    quant: str = "bf16"            # bf16 | fp8 | int4
    spec: str = "off"              # off | ngram
    spec_k: int = 4                # draft tokens proposed per step
    spec_accept: float = 0.6       # per-draft-token acceptance probability (ILLUSTRATIVE)
    time_scale: float = 10.0       # sim seconds per wall second (server only)
    seed: int = 0
    # Engine knobs (names follow vLLM flags)
    gpu_memory_utilization: float = 0.92   # vLLM default
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    enable_prefix_caching: bool = True
    # Modelling assumptions (all ILLUSTRATIVE, documented in the README)
    bw_efficiency: float = 0.80    # achieved / peak HBM bandwidth
    mfu: float = 0.50              # achieved / peak tensor FLOPs on large matmuls
    tp_efficiency: float = 0.90    # scaling loss per extra GPU in tensor parallel
    step_overhead_s: float = 0.003  # scheduler + kernel launch cost per engine step
    spec_overhead_s: float = 0.0005  # n-gram lookup and verify bookkeeping per step
    int4_compute_efficiency: float = 0.60  # dequantise-in-register penalty on the compute side
    activation_gb_per_gpu: float = 2.0     # CUDA context, activations, graphs

    def with_(self, **kw) -> "EngineConfig":
        return replace(self, **kw)

    # ---- derived quantities -------------------------------------------------
    @property
    def gpu_spec(self) -> GPUSpec:
        return GPUS[self.gpu]

    @property
    def model_spec(self) -> ModelSpec:
        return MODELS[self.model]

    @property
    def weight_bytes(self) -> float:
        return self.model_spec.params * QUANT_BYTES[self.quant]

    @property
    def kv_bytes_per_token(self) -> float:
        # KV cache stays 16-bit in every mode here (a deliberate simplification)
        return self.model_spec.kv_bytes_per_token(2.0)

    @property
    def tp_scale(self) -> float:
        n = self.num_gpus
        return n * (self.tp_efficiency ** (n - 1)) if n > 1 else 1.0

    @property
    def peak_flops(self) -> float:
        g = self.gpu_spec
        if self.quant == "fp8":
            return g.fp8_tflops * 1e12
        base = g.bf16_tflops * 1e12
        return base * self.int4_compute_efficiency if self.quant == "int4" else base

    @property
    def kv_capacity_blocks(self) -> int:
        g = self.gpu_spec
        usable = g.mem_gb * 1e9 * self.num_gpus * self.gpu_memory_utilization
        usable -= self.activation_gb_per_gpu * 1e9 * self.num_gpus
        free = usable - self.weight_bytes
        return max(0, int(free // (self.kv_bytes_per_token * BLOCK_SIZE)))

    def validate(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; choose from {sorted(MODELS)}")
        if self.gpu not in GPUS:
            raise ValueError(f"unknown gpu {self.gpu!r}; choose from {sorted(GPUS)}")
        if self.quant not in QUANT_BYTES:
            raise ValueError(f"unknown quant {self.quant!r}; choose from {sorted(QUANT_BYTES)}")
        if self.spec not in ("off", "ngram"):
            raise ValueError("spec must be 'off' or 'ngram'")
        if self.quant == "fp8" and self.gpu_spec.fp8_tflops <= 0:
            raise ValueError(f"{self.gpu_spec.name} has no FP8 tensor cores; use bf16 or int4")
        if self.kv_capacity_blocks < 64:
            need = self.weight_bytes / 1e9
            have = self.gpu_spec.mem_gb * self.num_gpus
            raise ValueError(
                f"{self.model_spec.name} {self.quant} needs {need:.0f} GB of weights; "
                f"{self.num_gpus}x {self.gpu_spec.name} gives {have:.0f} GB total and almost "
                f"nothing is left for KV cache. Add GPUs (--num-gpus) or quantise."
            )

    def describe(self) -> dict:
        return {
            "model": self.model_spec.name, "gpu": self.gpu_spec.name, "num_gpus": self.num_gpus,
            "quant": self.quant, "spec": self.spec, "spec_k": self.spec_k,
            "spec_accept": self.spec_accept, "kv_capacity_tokens": self.kv_capacity_blocks * BLOCK_SIZE,
            "weight_gb": round(self.weight_bytes / 1e9, 2), "seed": self.seed,
            "max_num_seqs": self.max_num_seqs, "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefix_caching": self.enable_prefix_caching,
        }
