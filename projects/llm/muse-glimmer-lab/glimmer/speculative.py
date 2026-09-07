"""DFlash, and why a diffusion drafter changes the speculative-decoding maths.

Speculative decoding is an old trick with a simple contract. A small cheap
model guesses the next few tokens, the big model checks all of them in one
forward pass, and you keep the longest correct prefix. The output is
*identical* to what the big model would have produced alone, because the big
model still gets the final say on every token. You are not trading quality for
speed. You are trading spare parallel compute for latency.

The reason it works at all is that generation is memory-bound. Reading 17 GB of
weights out of VRAM to produce one token wastes almost all of a modern GPU's
arithmetic. Checking sixteen candidate tokens costs barely more than checking
one, because it is the same weight read.

DFlash is the drafter Meta ships with Glimmer as `dflash-kquant.gguf`. The
technique comes from a paper by Chen, Liang and Liu at UC San Diego (ICML 2026,
arXiv:2602.06036), not from Meta, and Glimmer is an adoption of it. Worth
keeping straight.

The idea it replaces is autoregressive drafting, used by EAGLE-3 and almost
everything before it. A small transformer generates the guesses one at a time,
so guessing 16 tokens costs 16 sequential forward passes of the drafter. That
sequential cost is the ceiling. From the paper, average per-token latency is

    L = (T_draft + T_verify) / tau                                        (1)

where tau is the average number of tokens accepted per cycle. For an
autoregressive drafter, T_draft = gamma * t_step, so drafting cost grows
linearly with the number of guesses. Guess more and tau rises, but T_draft
rises just as fast, and the two roughly cancel. Published speedups sit around
2-3x for that reason.

DFlash makes the drafter a small block-diffusion model instead. It starts from
a block of 16 masked positions and denoises them all at once, so

    T_draft = t_parallel                                                  (2)

which barely moves as the block gets bigger. Once drafting cost stops scaling
with the number of guesses, the sensible move is to spend the savings on a
*deeper* drafter, and the paper does exactly that, using 5 layers where EAGLE-3
uses 1. A better drafter raises tau, and tau is now the only term that matters.
The paper reports up to 6.1x over plain decoding and about 2.5x over EAGLE-3.

One more design choice does the heavy lifting. A small model asked to guess
what a 30B model will say next, with nothing but the tokens so far, is guessing
from scratch. DFlash instead takes hidden states from five layers of the target
model, fuses them, and injects them into the key and value projections of every
draft layer. The drafter is conditioned on what the big model is already
thinking. The paper's ablation shows injecting at every layer beats injecting
once at the input, which is what EAGLE-3 does.

This module simulates equation (1) so you can turn the knobs yourself. It is a
latency model, not a real drafter. Real acceptance rates depend on the text.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured acceptance lengths from the DFlash paper, table 1, block size 16 at
# temperature 0. They anchor the simulator to something real rather than to a
# number chosen to make the graph look good.
PAPER_ACCEPTANCE = {
    "GSM8K": 6.54,
    "MATH-500": 7.87,
    "AIME25": 7.08,
    "HumanEval": 6.50,
    "MBPP": 5.95,
    "LiveCodeBench": 7.27,
    "MT-Bench": 4.24,
}


@dataclass
class Result:
    tokens_per_second: float
    speedup: float
    acceptance_length: float
    draft_ms: float
    verify_ms: float

    def __str__(self) -> str:
        return (
            f"{self.tokens_per_second:7.1f} tok/s  "
            f"{self.speedup:5.2f}x  "
            f"tau={self.acceptance_length:.2f}  "
            f"draft={self.draft_ms:.1f}ms verify={self.verify_ms:.1f}ms"
        )


def expected_acceptance(block_size: int, per_token_accept: float) -> float:
    """Expected accepted tokens per cycle, given a per-token acceptance rate.

    Verification stops at the first rejection, so a block only counts up to its
    first wrong token. If each token is independently correct with probability
    p, the expected run length before a rejection is

        sum over k=1..B of p^k

    and the target model always contributes one guaranteed bonus token on top,
    because whatever it sampled at the rejection point is itself valid output.
    That bonus is why tau is at least 1 and why speculative decoding can never
    be slower than a rounding error, only faster.

    This assumes independence, which is optimistic. Real rejections cluster,
    since one wrong token derails the tokens after it. The paper's measured tau
    of 6.5 at block 16 implies a p around 0.87, well below what independence
    would predict from their acceptance histograms.
    """
    if not 0.0 <= per_token_accept <= 1.0:
        raise ValueError("per_token_accept must be a probability")
    run = sum(per_token_accept**k for k in range(1, block_size + 1))
    return run + 1.0


def simulate(
    *,
    target_step_ms: float,
    block_size: int = 16,
    acceptance_length: float | None = None,
    per_token_accept: float = 0.87,
    drafter: str = "diffusion",
    draft_step_ms: float = 0.9,
    verify_overhead: float = 0.15,
) -> Result:
    """Average throughput under equation (1).

    `target_step_ms` is how long one forward pass of the 30B model takes. On an
    RTX 5090 at 4-bit, plain decoding runs about 74.9 tok/s, so roughly 13.3 ms.

    `drafter` selects the cost model:

      "none"           no drafting, plain autoregressive decoding
      "autoregressive" EAGLE-3 style, T_draft = block_size * draft_step_ms
      "diffusion"      DFlash style, T_draft = draft_step_ms, flat in block_size

    `verify_overhead` is the extra fraction of a forward pass that checking a
    whole block costs over checking one token. It is small precisely because
    the pass is memory-bound, and setting it to 0 is the idealisation that
    makes speculative decoding look better than it is.
    """
    if acceptance_length is None:
        acceptance_length = (
            1.0 if drafter == "none" else expected_acceptance(block_size, per_token_accept)
        )

    if drafter == "none":
        draft_ms = 0.0
        verify_ms = target_step_ms
        acceptance_length = 1.0
    elif drafter == "autoregressive":
        # Every guess is its own sequential pass. This is the term DFlash kills.
        draft_ms = block_size * draft_step_ms
        verify_ms = target_step_ms * (1 + verify_overhead)
    elif drafter == "diffusion":
        # One denoising pass produces the whole block, so block size is nearly
        # free. Deeper drafters cost more per pass, not more per token.
        draft_ms = draft_step_ms
        verify_ms = target_step_ms * (1 + verify_overhead)
    else:
        raise ValueError(f"unknown drafter {drafter!r}")

    per_token_ms = (draft_ms + verify_ms) / acceptance_length
    baseline_ms = target_step_ms

    return Result(
        tokens_per_second=1000.0 / per_token_ms,
        speedup=baseline_ms / per_token_ms,
        acceptance_length=acceptance_length,
        draft_ms=draft_ms,
        verify_ms=verify_ms,
    )
