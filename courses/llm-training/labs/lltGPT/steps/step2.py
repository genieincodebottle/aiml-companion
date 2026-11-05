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

torch.manual_seed(0)
x = torch.randn(2, 5, 768) * 3 + 1   # mean 1, std 3
for name, norm in [("LayerNorm", make_norm(GPT2)), ("RMSNorm", make_norm(MODERN))]:
    y = norm(x)
    n = sum(p.numel() for p in norm.parameters())
    print(f"{name:9}  mean {y.mean(-1).mean():.4f}  rms {y.pow(2).mean(-1).sqrt().mean():.4f}  parameters {n}")
