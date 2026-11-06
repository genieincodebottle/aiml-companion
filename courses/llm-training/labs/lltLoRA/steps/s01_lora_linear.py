import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle has 2 GPUs; this lab uses one
import math
import torch
import torch.nn as nn

torch.set_num_threads(4)

class LoRALinear(nn.Module):
    # y = W x + scale * B A x, with W frozen and only A and B trained.
    def __init__(self, base, r=16, alpha=32, rslora=False, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                      # freeze W (and its bias, if any)
        self.r = r
        self.scale = alpha / math.sqrt(r) if rslora else alpha / r
        w = base.weight
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=w.device, dtype=w.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # PEFT's default init for A; B stays 0
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y                                      # W already holds W + scale * B A
        return y + (self.drop(x) @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scale
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= (self.B @ self.A) * self.scale
            self.merged = False

torch.manual_seed(0)
base = nn.Linear(1024, 3584, bias=False)                 # the shape of one MLP up_proj in Qwen3.5-0.8B
x = torch.randn(4, 10, 1024)
y0 = base(x).detach()
layer = LoRALinear(base, r=16, alpha=32)
print("output unchanged at init:", torch.equal(layer(x), y0))
n_train = sum(p.numel() for p in layer.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
print(f"trainable {n_train:,} | frozen {n_frozen:,} | {100 * n_train / n_frozen:.2f}% of the layer")

layer(x).pow(2).mean().backward()
print("grad on W:", base.weight.grad, "| grad on A is all zero:", bool(layer.A.grad.abs().max() == 0),
      "| grad on B is all zero:", bool(layer.B.grad.abs().max() == 0))
for r in [1, 4, 16, 64, 256]:
    print(f"r = {r:>3}  alpha/r = {32 / r:7.4f}  rsLoRA alpha/sqrt(r) = {32 / math.sqrt(r):7.4f}")
