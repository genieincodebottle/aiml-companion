"""Experiment 4. Take the architecture apart and watch the memory move.

    uv run python scripts/04_kv_memory.py

No model needed, and no GPU. This is arithmetic over the model card.

Three tables:

  1. What the weights cost at each quantization.
  2. What the KV cache costs as context grows, with Glimmer's design and with
     each of its two memory decisions switched off.
  3. Whether a given card actually fits a given configuration.

The number to take away from table 2 is the last row. Turning off grouped-query
attention and the sliding window turns a 1.8 GiB cache into something no
consumer card can hold, and neither change touches the parameter count. This is
why "30B parameters" tells you almost nothing about whether a model fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets `python scripts/xx.py` work as well as `uv run python scripts/xx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glimmer.config import ARCH, QUANTS
from glimmer.memory import kv_cache_bytes, total_footprint_gib

GIB = 1024**3


def weights_table() -> None:
    print("\n1. Weights on disk\n" + "-" * 78)
    print(f"{'build':<18}{'bits':>6}{'GB':>8}   note")
    for name, spec in QUANTS.items():
        print(f"{name:<18}{spec['bits']:>6}{spec['file_gb']:>8.1f}   {spec['note']}")


def kv_table() -> None:
    contexts = [4_096, 32_768, 131_072, 262_144]
    print("\n2. KV cache, fp16, batch 1\n" + "-" * 78)
    print(f"{'configuration':<40}" + "".join(f"{c // 1024:>9}k" for c in contexts))

    rows = [
        ("Muse Glimmer as shipped", dict()),
        ("without the sliding window", dict(use_sliding_window=False)),
        ("without grouped-query attention", dict(use_gqa=False)),
        ("without either, a plain transformer", dict(use_gqa=False, use_sliding_window=False)),
    ]
    for label, kwargs in rows:
        cells = "".join(
            f"{kv_cache_bytes(c, **kwargs).total_gib:>9.1f}" for c in contexts
        )
        print(f"{label:<40}{cells}")

    shipped = kv_cache_bytes(131_072).total_gib
    plain = kv_cache_bytes(131_072, use_gqa=False, use_sliding_window=False).total_gib
    print(
        f"\n  At 131k the two decisions together save {plain - shipped:.0f} GiB, "
        f"a factor of {plain / shipped:.0f}."
    )

    breakdown = kv_cache_bytes(131_072)
    print(
        f"  Of the {shipped:.2f} GiB that remain, "
        f"{breakdown.global_bytes / GIB:.2f} GiB is the {breakdown.global_layers} global layers "
        f"and only {breakdown.local_bytes / GIB:.3f} GiB is the {breakdown.local_layers} local ones,\n"
        f"  because a local layer never stores more than its {ARCH.sliding_window}-token window."
    )


def fit_table() -> None:
    print("\n3. Does it fit\n" + "-" * 78)
    cards = [("RTX 4090 / 24GB", 24), ("RTX 5090 / 32GB", 32), ("Mac 48GB unified", 48)]
    configs = [
        ("kquant-17gb, 32k ctx", dict(quant="kquant-17gb", context=32_768)),
        ("kquant-17gb, 131k ctx", dict(quant="kquant-17gb", context=131_072)),
        ("kquant-17gb, 131k + vision + drafter",
         dict(quant="kquant-17gb", context=131_072, with_vision=True, with_drafter=True)),
        ("kquant-dynamic, 131k + drafter",
         dict(quant="kquant-dynamic", context=131_072, with_drafter=True)),
    ]
    print(f"{'configuration':<40}{'needs':>8}   " + "  ".join(f"{n:<17}" for n, _ in cards))
    for label, kwargs in configs:
        need = total_footprint_gib(**kwargs)["total"]
        marks = "  ".join(
            f"{('fits' if need <= cap else 'no'):<17}" for _, cap in cards
        )
        print(f"{label:<40}{need:>8.1f}   {marks}")

    print(
        "\n  The vision encoder and the DFlash drafter are 3 GB together, and both are\n"
        "  easy to forget when sizing a card. On 24 GB they are the difference between\n"
        "  the two 4-bit builds: kquant-17gb still fits with both loaded, and\n"
        "  kquant-dynamic does not fit with even one of them."
    )


if __name__ == "__main__":
    weights_table()
    kv_table()
    fit_table()
