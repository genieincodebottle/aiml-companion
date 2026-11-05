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

for d in [192, 320, 768, 4096]:
    h = swiglu_hidden(d)
    print(f"d = {d:>4}  8d/3 = {8 * d / 3:7.1f}  hidden {h:>5}  MLP weights {3 * d * h:>11,}  GELU 4d MLP {8 * d * d:>11,}")
