import math

def swiglu_hidden(d):
    return 64 * math.ceil(8 * d / 3 / 64)

def count_params(vocab, ctx, d, layers, kind):
    emb = vocab * d                            # token embedding, shared with the head
    if kind == "gpt2":
        per_layer = 12 * d * d + 13 * d        # qkv 3d^2+3d, proj d^2+d, MLP 8d^2+5d, two LayerNorms 4d
        return emb + ctx * d + layers * per_layer + 2 * d
    h = swiglu_hidden(d)
    per_layer = 4 * d * d + 3 * d * h + 2 * d  # qkv and proj 4d^2, gate, up and down 3dh, two RMSNorms 2d
    return emb + layers * per_layer + d

assert count_params(50257, 1024, 768, 12, "gpt2") == 124_439_808, "GPT-2 124M should be 124,439,808"
assert count_params(50257, 1024, 768, 12, "modern") == 123_551_232, "modern config at GPT-2 size should be 123,551,232"
assert count_params(4096, 512, 320, 8, "modern") == 11_474_240, "TinyStories model should be 11,474,240; check the SwiGLU rounding"
assert count_params(2048, 128, 192, 4, "gpt2") == 2_197_632, "step 8 gpt2 variant should be 2,197,632"
n = count_params(50257, 1024, 768, 12, "gpt2")
print("GPT-2 124M", f"{n:,}", "| 12Ld^2 + Vd", f"{12 * 12 * 768 ** 2 + 50257 * 768:,}", "| embedding share", f"{50257 * 768 / n:.0%}")
print("All checks passed")
