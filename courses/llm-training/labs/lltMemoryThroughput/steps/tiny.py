# tiny.py - the lltTinyStories GPT (RMSNorm, RoPE, SwiGLU, tied embeddings) with two fixes and two probes.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_tables(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, d, h, qk_norm):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3 / 64 + 0.999) * 64
        self.w1, self.w3, self.w2 = nn.Linear(d, hid, bias=False), nn.Linear(d, hid, bias=False), nn.Linear(hid, d, bias=False)
        self.qn = RMSNorm(d // h) if qk_norm else None      # QK-norm, one RMSNorm over each head's queries
        self.kn = RMSNorm(d // h) if qk_norm else None      # and one over its keys
        self.max_logit = None

    def forward(self, x, cos, sin, probe=False):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if probe:   # largest attention logit before softmax, computed in fp32
            s = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            self.max_logit = s.masked_fill(causal, float("-inf")).max().item()
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GPT(nn.Module):
    def __init__(self, V=4096, d=320, L=8, H=8, qk_norm=False):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, H, qk_norm) for _ in range(L)])
        self.norm, self.head, self.H = RMSNorm(d), nn.Linear(d, V, bias=False), H
        self.head.weight = self.emb.weight
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * L) if n.endswith(("o.weight", "w2.weight")) else 0.02)

    def forward(self, idx, targets, z_loss=0.0, probe=False):
        cos, sin = rope_tables(idx.shape[1], self.emb.embedding_dim // self.H, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin, probe)
        logits = self.head(self.norm(x)).float()
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if z_loss or probe:
            log_z = torch.logsumexp(logits, dim=-1)          # log of the softmax normalizer, per token
            if z_loss:
                loss = loss + z_loss * log_z.pow(2).mean()  # z-loss pulls log Z toward 0
            if probe:
                self.stats = {"maxAttnLogit": max(b.max_logit for b in self.blocks),
                              "meanLogZ": log_z.mean().item(), "maxAbsLogit": logits.abs().max().item()}
        return loss
