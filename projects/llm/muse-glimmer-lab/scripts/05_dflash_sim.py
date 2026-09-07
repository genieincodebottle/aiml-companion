"""Experiment 5. Why a diffusion drafter beats an autoregressive one.

    uv run python scripts/05_dflash_sim.py

A latency simulation of the DFlash paper's equation (1), so you can see the
term that matters rather than take the speedup on faith.

    L = (T_draft + T_verify) / tau

Three tables:

  1. Plain decoding against both drafter styles, at RTX 5090 speeds.
  2. What happens to each as the block size grows. This is the whole argument.
  3. The paper's measured acceptance lengths, turned into predicted speedups.

Table 2 is the one to read carefully. The autoregressive drafter improves and
then gets worse, because every extra guess costs it another sequential forward
pass. The diffusion drafter keeps improving, because sixteen guesses cost it
the same single pass that one guess did.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Lets `python scripts/xx.py` work as well as `uv run python scripts/xx.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glimmer.speculative import PAPER_ACCEPTANCE, expected_acceptance, simulate

# Meta reports plain decoding at 74.9 tok/s on an RTX 5090 for the 4-bit build,
# rising to 233.4 with DFlash, a 3.1x speedup. 74.9 tok/s is 13.35 ms a token.
TARGET_STEP_MS = 1000 / 74.9


def head_to_head() -> None:
    print("\n1. One block of 16, RTX 5090 speeds\n" + "-" * 78)
    for drafter in ("none", "autoregressive", "diffusion"):
        result = simulate(target_step_ms=TARGET_STEP_MS, drafter=drafter, block_size=16)
        print(f"  {drafter:<16}{result}")
    print(
        "\n  Meta measures 74.9 -> 233.4 tok/s on this card, a 3.1x speedup. The\n"
        "  simulation says 5.7x, so read it as the shape of the argument and not as a\n"
        "  prediction. It is idealised in three ways that all flatter DFlash: it\n"
        "  assumes rejections are independent when in practice they cluster, it prices\n"
        "  block verification at a flat 15% over a single token, and it ignores\n"
        "  sampling, detokenization and the runtime's own per-step overhead. The gap\n"
        "  between 5.7x and 3.1x is roughly what those three cost you in reality."
    )


def block_size_sweep() -> None:
    print("\n2. Speedup against block size\n" + "-" * 78)
    print(f"{'block':>7}{'tau':>8}{'autoregressive':>18}{'diffusion':>13}")
    for block in (1, 2, 4, 8, 16, 32, 64):
        tau = expected_acceptance(block, 0.87)
        ar = simulate(target_step_ms=TARGET_STEP_MS, drafter="autoregressive", block_size=block)
        df = simulate(target_step_ms=TARGET_STEP_MS, drafter="diffusion", block_size=block)
        print(f"{block:>7}{tau:>8.2f}{ar.speedup:>17.2f}x{df.speedup:>12.2f}x")
    print(
        "\n  The autoregressive column peaks and turns over. Its drafting cost grows\n"
        "  with the block while acceptance flattens out, so past the peak every extra\n"
        "  guess costs more than it returns. The diffusion column keeps climbing\n"
        "  because its drafting cost does not depend on the block size.\n"
        "  Meta ships block size 16, which is where the real curve pays best."
    )


def from_paper() -> None:
    print("\n3. Predicted speedup from the paper's measured acceptance\n" + "-" * 78)
    print(f"{'benchmark':<18}{'tau (measured)':>16}{'predicted':>12}")
    for name, tau in PAPER_ACCEPTANCE.items():
        result = simulate(
            target_step_ms=TARGET_STEP_MS, drafter="diffusion", acceptance_length=tau
        )
        print(f"{name:<18}{tau:>16.2f}{result.speedup:>11.2f}x")
    print(
        "\n  Chat accepts fewer tokens than maths or code. Prose has more valid\n"
        "  continuations at every position, so the drafter is wrong sooner. Agentic\n"
        "  work looks more like the code rows, which is the case Glimmer cares about."
    )


if __name__ == "__main__":
    head_to_head()
    block_size_sweep()
    from_paper()
    print(
        "\nNone of this changes the output. The target model verifies every token,\n"
        "so speculative decoding is lossless. You are spending idle parallel compute\n"
        "to avoid re-reading 17 GB of weights for each token.\n"
    )
