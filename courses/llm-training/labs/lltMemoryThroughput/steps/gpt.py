# gpt.py - GPT-2 small (124M) in plain PyTorch, with the two memory switches this lab measures.
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab: int = 50257
    ctx: int = 1024
    layers: int = 12
    heads: int = 12
    d: int = 768


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.fc, self.out = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


def chunk_loss(h, w, y):
    return F.cross_entropy((h @ w.t()).float(), y, reduction="sum")


class GPT(nn.Module):
    def __init__(self, c, ckpt=False, chunks=0):
        super().__init__()
        self.ckpt, self.chunks = ckpt, chunks   # activation checkpointing, chunked cross-entropy
        self.wte, self.wpe = nn.Embedding(c.vocab, c.d), nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.wte.weight          # tied, as in GPT-2
        for n, p in self.named_parameters():
            if p.dim() == 2:
                small = n.endswith(("proj.weight", "out.weight"))
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.layers) if small else 0.02)
            elif n.endswith("bias"):
                nn.init.zeros_(p)

    def forward(self, idx, targets):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
        h, y = self.ln_f(x).flatten(0, 1), targets.flatten()
        if not self.chunks:
            return F.cross_entropy(self.head(h).float(), y)
        # Chunked cross-entropy: each chunk's logits are recomputed in backward, so one chunk is alive at a time
        parts = [checkpoint(chunk_loss, hc, self.head.weight, yc, use_reentrant=False)
                 for hc, yc in zip(h.chunk(self.chunks), y.chunk(self.chunks))]
        return sum(parts) / y.numel()
