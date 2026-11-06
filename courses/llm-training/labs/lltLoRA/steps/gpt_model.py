# gpt_model.py - the GPT from the lab "Assemble a Modern GPT" (steps 2 to 4), definitions only.
import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab: int = 50257
    ctx: int = 1024
    d: int = 768
    layers: int = 12
    heads: int = 12
    norm: str = "rmsnorm"   # "rmsnorm" or "layernorm"
    mlp: str = "swiglu"     # "swiglu" or "gelu"
    pos: str = "rope"       # "rope" or "learned"
    prenorm: bool = True
    bias: bool = False
    eps: float = 1e-5

GPT2 = Config(norm="layernorm", mlp="gelu", pos="learned", bias=True)
MODERN = Config()

class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def make_norm(cfg):
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d, cfg.eps)
    return nn.LayerNorm(cfg.d, eps=cfg.eps, bias=cfg.bias)


def rope_tables(T, hd, offset=0, base=10000.0, device=None):
    inv = base ** (-torch.arange(0, hd, 2, device=device).float() / hd)
    ang = torch.outer(torch.arange(offset, offset + T, device=device).float(), inv)
    return ang.cos(), ang.sin()

def apply_rope(x, cos, sin):
    # x is (batch, heads, T, head_dim). Rotate the pairs (x[2i], x[2i+1]) as in step 1.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.rope = cfg.heads, cfg.pos == "rope"
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=cfg.bias)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # queries and keys, never values
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

def swiglu_hidden(d):
    # 8d/3 keeps the parameter count of a 4d GELU MLP; round up to a multiple of 64
    return 64 * math.ceil(8 * d / 3 / 64)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.swiglu = cfg.mlp == "swiglu"
        hidden = swiglu_hidden(cfg.d) if self.swiglu else 4 * cfg.d
        self.gate = nn.Linear(cfg.d, hidden, bias=cfg.bias) if self.swiglu else None
        self.up = nn.Linear(cfg.d, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.d, bias=cfg.bias)

    def forward(self, x):
        if self.swiglu:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        return self.down(F.gelu(self.up(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.prenorm = cfg.prenorm
        self.n1, self.n2 = make_norm(cfg), make_norm(cfg)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos=None, sin=None):
        if self.prenorm:
            x = x + self.attn(self.n1(x), cos, sin)
            return x + self.mlp(self.n2(x))
        x = self.n1(x + self.attn(x, cos, sin))
        return self.n2(x + self.mlp(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d) if cfg.pos == "learned" else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = make_norm(cfg)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight   # tie first, so the init below sets the shared tensor once
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                # the two layers per block that write into the residual stream
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.layers))
            elif p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, offset=0):
        B, T = idx.shape
        x = self.tok(idx)
        cos = sin = None
        if self.pos is None:
            cos, sin = rope_tables(T, self.cfg.d // self.cfg.heads, offset, device=idx.device)
        else:
            x = x + self.pos(torch.arange(offset, offset + T, device=idx.device))
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.norm(x))

# The model our TinyStories pretraining run uses (lltTinyStories)
TINY = replace(MODERN, vocab=4096, ctx=512, d=320, layers=8, heads=8)
