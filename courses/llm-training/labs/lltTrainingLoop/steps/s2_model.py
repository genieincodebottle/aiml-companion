import math
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d, bias=False), nn.LayerNorm(d, bias=False)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc, self.out = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        p = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, vocab, block, n_layer, n_head, d, dropout=0.0):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, n_head, dropout) for _ in range(n_layer)])
        self.ln_f, self.head = nn.LayerNorm(d, bias=False), nn.Linear(d, vocab, bias=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                # GPT-2 init: std 0.02, and the residual output layers scaled by 1/sqrt(2L)
                std = 0.02 / math.sqrt(2 * n_layer) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):
        x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def param_groups(model, weight_decay):
    # Decay the weight matrices of linear layers only. Norm gains and embeddings keep their size.
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if isinstance(module, nn.Linear) and name == "weight":
                decay.append(p)
            else:
                no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
