"""Where the memory actually goes, and why Glimmer fits on one card.

Weights are the number everyone quotes and the KV cache is the number that
makes people run out of memory at 3% of the way through a long agentic run.
This module computes both, and lets you switch off Glimmer's two memory
decisions one at a time so you can see what each one buys.

The two decisions:

  Grouped-query attention   32 query heads share 2 key/value heads. The cache
                            stores keys and values, not queries, so the cache
                            shrinks by 16x against a model that gives every
                            query head its own pair.

  Interleaved local/global  Three sliding-window layers then one full-attention
                            layer, repeating. Only the 13 global layers hold
                            the whole context. The other 39 hold 2,048 tokens
                            and forget the rest.

Run scripts/04_kv_memory.py to see the table. The short version is that the
combination is worth about 110 GB at a 131,072-token context, which is the
difference between a laptop and a server rack.

Every architecture constant comes from config.ARCH, which cites the model card.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ARCH, QUANTS, Arch

GIB = 1024**3


@dataclass
class KVBreakdown:
    global_layers: int
    local_layers: int
    global_bytes: int
    local_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.global_bytes + self.local_bytes

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


def kv_cache_bytes(
    context: int,
    arch: Arch = ARCH,
    *,
    bytes_per_value: int = 2,
    use_gqa: bool = True,
    use_sliding_window: bool = True,
    batch: int = 1,
) -> KVBreakdown:
    """Bytes of key/value cache for `context` tokens.

    The per-layer, per-token cost is

        2 x kv_heads x head_dim x bytes_per_value

    The leading 2 is the key and the value. `bytes_per_value` is 2 for the
    usual fp16 cache and 1 if you quantize the cache to 8 bits, which llama.cpp
    exposes as `--cache-type-k q8_0 --cache-type-v q8_0`.

    Set `use_gqa=False` to bill every query head its own key/value pair, which
    is what multi-head attention does. Set `use_sliding_window=False` to make
    every layer store the full context, which is what a conventional
    all-global transformer does. Those two switches are the experiment.
    """
    kv_heads = arch.kv_heads if use_gqa else arch.q_heads
    per_token_per_layer = 2 * kv_heads * arch.head_dim * bytes_per_value

    # Count the layers by walking them, rather than multiplying whole periods
    # and assuming the remainder is all local. That shortcut happens to be
    # right for Glimmer -- 52 layers is exactly 13 periods of (l,l,l,g) -- and
    # silently wrong the moment either number changes. With the global layer
    # first in the pattern, 54 layers has 14 global layers and the arithmetic
    # version returns 13, understating the cache by a full layer.
    pattern = arch.attention_pattern
    kinds = [pattern[i % len(pattern)] for i in range(arch.layers)]
    n_global = kinds.count("global")
    n_local = arch.layers - n_global

    if not use_sliding_window:
        # Every layer becomes a full-attention layer.
        n_global, n_local = arch.layers, 0

    # A sliding-window layer never stores more than its window, however long
    # the conversation gets. That cap is the whole point of the design.
    local_span = min(context, arch.sliding_window)

    return KVBreakdown(
        global_layers=n_global,
        local_layers=n_local,
        global_bytes=n_global * context * per_token_per_layer * batch,
        local_bytes=n_local * local_span * per_token_per_layer * batch,
    )


def weight_bytes(quant: str = "kquant-17gb") -> int:
    """On-disk size of the weights for a named build, from the GGUF repo."""
    if quant not in QUANTS:
        raise KeyError(f"unknown quant {quant!r}, expected one of {sorted(QUANTS)}")
    return int(QUANTS[quant]["file_gb"] * GIB)


def total_footprint_gib(
    context: int,
    quant: str = "kquant-17gb",
    *,
    with_vision: bool = False,
    with_drafter: bool = False,
    overhead_gib: float = 1.0,
    **kv_kwargs,
) -> dict[str, float]:
    """A realistic VRAM estimate, itemised.

    `overhead_gib` covers the CUDA context, activations and the runtime's own
    buffers. One gigabyte is a reasonable llama.cpp figure at these batch
    sizes; vLLM reserves far more up front by design, so trust its own
    reporting over this number when you serve with vLLM.

    The vision and drafter files are optional and each is easy to forget when
    sizing a card. Together they are 3 GB, which is exactly enough to turn a
    working 24 GB configuration into an out-of-memory error.
    """
    kv = kv_cache_bytes(context, **kv_kwargs)
    items = {
        "weights": weight_bytes(quant) / GIB,
        "kv_cache": kv.total_gib,
        "overhead": overhead_gib,
    }
    if with_vision:
        items["vision_encoder"] = 1.4  # mmproj-kquant.gguf
    if with_drafter:
        items["dflash_drafter"] = 1.6  # dflash-kquant.gguf
    items["total"] = sum(items.values())
    return items
