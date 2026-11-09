import math
import torch
from trl.kernels.chunked_divergence import ChunkedDivergenceFunction   # the kernel inside TRL's DistillationTrainer

def divergence(student_logits, teacher_logits, beta):
    # Per-token divergence over the whole vocabulary. beta 0: forward KL(teacher || student).
    # beta 1: reverse KL(student || teacher). In between: generalized JSD with mixture m = beta * teacher + (1 - beta) * student.
    lps, lpt = student_logits.float().log_softmax(-1), teacher_logits.float().log_softmax(-1)
    if beta == 0:
        return (lpt.exp() * (lpt - lps)).sum(-1)
    if beta == 1:
        return (lps.exp() * (lps - lpt)).sum(-1)
    m = torch.logsumexp(torch.stack([lps + math.log(1 - beta), lpt + math.log(beta)]), 0)
    return beta * (lpt.exp() * (lpt - m)).sum(-1) + (1 - beta) * (lps.exp() * (lps - m)).sum(-1)

# TRL never builds the [tokens, vocab] logits. It takes hidden states and each model's output head and works in tiles.
torch.manual_seed(0)
V, N = 151936, 6                                  # Qwen3's vocabulary size, 6 token positions
h_s, h_t = torch.randn(N, 64), torch.randn(N, 96) # student and teacher hidden sizes differ; only the vocabulary is shared
W_s, W_t = torch.randn(V, 64) * 0.1, torch.randn(V, 96) * 0.1
for beta in (0.0, 0.5, 1.0):
    a, b = h_s.clone().requires_grad_(True), h_s.clone().requires_grad_(True)
    ours = divergence(a @ W_s.T, h_t @ W_t.T, beta)
    ours.sum().backward()
    theirs, _ = ChunkedDivergenceFunction.apply(b, W_s, None, h_t, W_t, None, beta)
    theirs.sum().backward()
    print("beta", beta, "| ours", round(ours.mean().item(), 6), "| TRL", round(theirs.mean().item(), 6),
          "| largest gap", "%.1e" % float((ours - theirs).detach().abs().max()),
          "| largest gradient gap", "%.1e" % float((a.grad - b.grad).abs().max()))