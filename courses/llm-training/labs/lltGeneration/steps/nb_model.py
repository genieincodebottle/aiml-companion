import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT from lltGPT and lltTinyStories: pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings.
# One addition: n_kv_heads, for grouped-query attention in step 8 (equal to n_heads by default).

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)

def rope_tables(start, T, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(start, start + T).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)

class Block(nn.Module):
    def __init__(self, d, h, h_kv):
        super().__init__()
        self.h, self.h_kv, self.hd = h, h_kv, d // h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, d + 2 * h_kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3 / 64 + 0.999) * 64
        self.w1 = nn.Linear(d, hid, bias=False)
        self.w3 = nn.Linear(d, hid, bias=False)
        self.w2 = nn.Linear(hid, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split([C, self.h_kv * self.hd, self.h_kv * self.hd], dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h_kv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.h // self.h_kv, dim=1)
        v = v.repeat_interleave(self.h // self.h_kv, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

class GPT(nn.Module):
    def __init__(self, V, d, L, H, ctx, H_kv=None):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H, H_kv or H) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        self.H, self.ctx = H, ctx
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(("o.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
    def forward(self, idx):
        cos, sin = rope_tables(0, idx.shape[1], self.emb.embedding_dim // self.H)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))
