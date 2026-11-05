# Peak memory of one fp16 autocast + fused AdamW step of GPT-2 124M, from the tensors autograd keeps.
N, L, d, V = 124_439_808, 12, 768, 50257
GB = 1e9

def predict_peak_gb(b, s, ckpt=False, chunks=0):
    tokens = b * s
    layer = 4 * d if ckpt else 38 * d     # bytes per token per block kept for backward
    head = 8 * d if chunks else 6 * d     # final LayerNorm input, plus the hidden state the head reads
    logits = 12 * V / max(chunks, 1)      # fp32 log-probs, their gradient and the logits gradient
    during_loss = 14 * N + tokens * (L * layer + head + logits)   # params 4, Adam 8, fp16 weight copies 2
    end_of_backward = 16 * N + 4 * V * d     # params 4, grads 4, Adam 8, plus the dense embedding gradient
    return max(during_loss, end_of_backward) / GB

for b, s in [(1, 256), (8, 1024), (12, 1024), (16, 1024)]:
    print(f"b={b:>2} s={s:>4}  predicted peak {predict_peak_gb(b, s):5.2f} GB")
print(f"per token: blocks {L * 38 * d:,} B, logits {12 * V:,} B")
