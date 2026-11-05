from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

T, cfg = 1024, GPT2
model = GPT(cfg).eval()
# The counter sees attention only through the plain-math backend, so force it for this count
with FlopCounterMode(display=False) as counter, torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    model(torch.zeros(1, T, dtype=torch.long))
counted = counter.get_total_flops() / T
weights = 12 * cfg.layers * cfg.d ** 2 + cfg.vocab * cfg.d   # every weight in a matmul, the head included
attention = 4 * cfg.layers * T * cfg.d                        # scores Q K^T, then the weighted sum of V
print(f"forward FLOPs per token at T = {T}  counted {counted / 1e6:.1f}M  formula 2 x weights + attention = {(2 * weights + attention) / 1e6:.1f}M")
print(f"attention share {attention / (2 * weights + attention):.1%}  training is about 3x forward, {3 * (2 * weights + attention) / 1e6:.0f}M per token")
