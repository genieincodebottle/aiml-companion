# gpt.py - the course GPT (lltGPT: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings) at capstone sizes.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CONFIGS = {
    "124m": dict(V=50257, d=768, L=12, H=12, ctx=1024),   # GPT-2 small shape
    "40m": dict(V=50257, d=384, L=12, H=6, ctx=1024),     # same depth, half the width
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_tables(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3 / 64 + 0.999) * 64
        self.w1 = nn.Linear(d, hid, bias=False)
        self.w3 = nn.Linear(d, hid, bias=False)
        self.w2 = nn.Linear(hid, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GPT(nn.Module):
    def __init__(self, V, d, L, H, ctx, ckpt=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight            # tied, as in GPT-2
        self.H, self.ctx, self.ckpt = H, ctx, ckpt
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(("o.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        cos, sin = rope_tables(T, self.emb.embedding_dim // self.H, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = checkpoint(b, x, cos, sin, use_reentrant=False) if self.ckpt else b(x, cos, sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits                                  # used by generation and the Hugging Face wrapper
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())


def flops_per_token(cfg, n_params):
    # nanoGPT / PaLM convention: 6N for the matmuls plus 12*L*d*T for attention scores and values
    return 6 * n_params + 12 * cfg["L"] * cfg["d"] * cfg["ctx"]
