"""The lab model for lltGeneration: the same GPT as kaggle/cells_tinystories.py, plus a KV cache.

Architecture (unchanged from the TinyStories pretraining cells): pre-norm RMSNorm, RoPE on
interleaved pairs, SwiGLU, fused QKV projection, tied input and output embeddings.

Added for this lab:
  n_kv_heads   grouped-query attention (GQA). n_kv_heads == n_heads is ordinary multi-head attention.
  KVCache      preallocated key and value tensors, one slot per layer, plus the write position.
  forward(idx, cache)   with a cache, the new tokens get RoPE positions cache.pos .. cache.pos + T - 1,
               their keys and values are written into the cache, and attention reads every cached slot
               up to the last new token, under a causal mask that is offset by cache.pos.

`bug` exists only so measure.py can reproduce the two broken runs of the page:
  "rope_offset"   RoPE positions restart at 0 for every forward call (the offset is forgotten)
  "prefill_mask"  no causal mask when more than one token is processed with a cache (the prefill chunk)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_tables(start, T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(start, start + T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class KVCache:
    """Preallocated keys and values: shape (layers, batch, kv_heads, max_len, head_dim) each."""

    def __init__(self, n_layers, batch, n_kv_heads, max_len, head_dim, dtype=torch.float32, device="cpu"):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.pos = 0

    def nbytes(self):
        return self.k.nbytes + self.v.nbytes


class Block(nn.Module):
    def __init__(self, d, h, h_kv, mlp=True):
        super().__init__()
        assert h % h_kv == 0
        self.mlp = mlp
        self.h, self.h_kv, self.hd = h, h_kv, d // h
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, d + 2 * h_kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if mlp:
            hid = int(8 * d / 3 / 64 + 0.999) * 64
            self.n2 = RMSNorm(d)
            self.w1 = nn.Linear(d, hid, bias=False)
            self.w3 = nn.Linear(d, hid, bias=False)
            self.w2 = nn.Linear(hid, d, bias=False)

    def forward(self, x, cos, sin, cache=None, layer=0, bug=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split([C, self.h_kv * self.hd, self.h_kv * self.hd], dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h_kv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            mask, causal = None, True
        else:
            p = cache.pos
            cache.k[layer, :, :, p:p + T] = k
            cache.v[layer, :, :, p:p + T] = v
            k, v = cache.k[layer, :, :, :p + T], cache.v[layer, :, :, :p + T]
            # Query i sits at position p + i and may read keys 0 .. p + i.
            mask = torch.arange(p + T, device=x.device)[None, :] <= (p + torch.arange(T, device=x.device))[:, None]
            if bug == "prefill_mask" and T > 1:
                mask = None
            causal = False
        if self.h_kv != self.h:
            k = k.repeat_interleave(self.h // self.h_kv, dim=1)
            v = v.repeat_interleave(self.h // self.h_kv, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        if not self.mlp:
            return x
        hh = self.n2(x)
        return x + self.w2(F.silu(self.w1(hh)) * self.w3(hh))


class GPT(nn.Module):
    def __init__(self, V, d, L, H, ctx, H_kv=None, mlp=True):
        super().__init__()
        self.cfg = dict(V=V, d=d, L=L, H=H, ctx=ctx, H_kv=H_kv or H, mlp=mlp)
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H, H_kv or H, mlp) for _ in range(L)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.emb.weight
        self.H, self.ctx = H, ctx
        self.bug = None
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(("o.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def new_cache(self, batch, max_len, dtype=torch.float32):
        c = self.cfg
        return KVCache(c["L"], batch, c["H_kv"], max_len, c["d"] // c["H"], dtype, self.emb.weight.device)

    def forward(self, idx, cache=None):
        T = idx.shape[1]
        start = 0 if (cache is None or self.bug == "rope_offset") else cache.pos
        cos, sin = rope_tables(start, T, self.emb.embedding_dim // self.H, idx.device)
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, cache, i, self.bug)
        if cache is not None:
            cache.pos += T
        return self.head(self.norm(x))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = GPT(c["V"], c["d"], c["L"], c["H"], c["ctx"], c.get("H_kv"), c.get("mlp", True))
    m.load_state_dict(ck["model"])
    return m.eval(), ck
