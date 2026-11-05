import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, C, h, dropout=0.0):
        super().__init__()
        assert C % h == 0, "the width C must split evenly into h heads"
        self.h, self.d, self.p = h, C // h, dropout
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)

    def attend(self, q, k, v):                                    # (B, h, T, d) each
        T = q.shape[-2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)      # (B, h, T, T)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        w = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        return F.dropout(w, self.p, training=self.training) @ v   # (B, h, T, d)

    def forward(self, x):                                         # (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                    # (B, T, C) each
        q, k, v = (a.view(B, T, self.h, self.d).transpose(1, 2) for a in (q, k, v))
        y = self.attend(q, k, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class SDPAAttention(CausalSelfAttention):                         # same weights, PyTorch's kernel
    def attend(self, q, k, v):
        p = self.p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)

torch.manual_seed(0)
torch.set_num_threads(4)
B, T, C, h = 4, 128, 256, 8
x = torch.randn(B, T, C)
for dtype, tol in [(torch.float32, 1e-5), (torch.float64, 1e-12)]:
    ours = CausalSelfAttention(C, h).to(dtype).eval()
    ref = SDPAAttention(C, h).to(dtype).eval()
    ref.load_state_dict(ours.state_dict())
    with torch.no_grad():
        a, b = ours(x.to(dtype)), ref(x.to(dtype))
    print(str(dtype).ljust(14), "max abs error", f"{(a - b).abs().max().item():.2e}",
          "| allclose", torch.allclose(a, b, rtol=0, atol=tol))
print("parameters", sum(p.numel() for p in ours.parameters()), "= 4 x C x C =", 4 * C * C)
