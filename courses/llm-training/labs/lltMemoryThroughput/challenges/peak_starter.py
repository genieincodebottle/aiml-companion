import numpy as np

def predict_peak_gb(N, L, d, V, b, s, ckpt=False, chunks=0):
    # TODO: peak allocated GB of one fp16 autocast + fused AdamW step of a GPT-2 style model.
    # Two candidates, take the larger:
    #   during the loss backward: 14 bytes per parameter, plus per token the block activations
    #     (38d per block, or 4d with ckpt), the head (6d, or 8d with chunks) and the logits (12V, divided by chunks)
    #   at the end of backward: 16 bytes per parameter plus one dense embedding gradient, 4Vd
    return None

# Checks. Run the cell. It prints All checks passed when every prediction is within 5% of our T4 runs.
# Measured on a Kaggle T4, runs lltMemoryThroughput-T2-memory-grid and -memory-cuts.
gpt2 = dict(N=124_439_808, L=12, d=768, V=50257)
measured = [
    ((1, 256), {}, 2.174), ((4, 512), {}, 3.700), ((16, 256), {}, 5.631), ((8, 1024), {}, 9.477),
    ((12, 1024), {}, 13.323), ((8, 1024), {"ckpt": True}, 6.880), ((8, 1024), {"chunks": 8}, 5.348),
    ((8, 1024), {"ckpt": True, "chunks": 8}, 2.750),
]
worst = 0.0
for (b, s), sw, got in measured:
    pred = predict_peak_gb(**gpt2, b=b, s=s, **sw)
    assert pred is not None, "predict_peak_gb returns None"
    err = pred / got - 1
    worst = max(worst, abs(err))
    assert abs(err) < 0.05, f"b={b} s={s} {sw}: predicted {pred:.2f} GB, measured {got} GB"
print(f"largest error {worst:.1%}")
fits = [b for b in (8, 12, 16) if predict_peak_gb(**gpt2, b=b, s=1024) < 15.6]
assert fits == [8, 12], "at s=1024 batches 8 and 12 fit on a 15.6 GB T4 and 16 does not"
print("All checks passed")
