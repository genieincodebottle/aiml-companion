import math

def swiglu_hidden(d):
    # 8d/3, rounded up to a multiple of 64, as in the lab's MLP
    return 64 * math.ceil(8 * d / 3 / 64)

def count_params(vocab, ctx, d, layers, kind):
    # TODO: the exact parameter count of the lab's GPT, head tied to the embedding.
    # kind "gpt2": LayerNorm with bias, GELU MLP of width 4d, biases on every linear layer,
    #              learned positions (ctx x d).
    # kind "modern": RMSNorm (weight only), SwiGLU MLP with three matrices of width
    #                swiglu_hidden(d), no biases, RoPE (no position parameters).
    return None

# Checks. Run the cell. It prints All checks passed when count_params is right.
assert count_params(50257, 1024, 768, 12, "gpt2") is not None, "count_params returns None"
assert count_params(50257, 1024, 768, 12, "gpt2") == 124_439_808, "GPT-2 124M should be 124,439,808"
assert count_params(50257, 1024, 768, 12, "modern") == 123_551_232, "modern config at GPT-2 size should be 123,551,232"
assert count_params(4096, 512, 320, 8, "modern") == 11_474_240, "TinyStories model should be 11,474,240; check the SwiGLU rounding"
assert count_params(2048, 128, 192, 4, "gpt2") == 2_197_632, "step 8 gpt2 variant should be 2,197,632"
n = count_params(50257, 1024, 768, 12, "gpt2")
print("GPT-2 124M", f"{n:,}", "| 12Ld^2 + Vd", f"{12 * 12 * 768 ** 2 + 50257 * 768:,}", "| embedding share", f"{50257 * 768 / n:.0%}")
print("All checks passed")
