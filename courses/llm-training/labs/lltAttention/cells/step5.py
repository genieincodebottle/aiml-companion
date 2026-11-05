# Uses CausalSelfAttention, SDPAAttention and C from step 4. Run this cell after it.
def leakage(attn, T=32, seed=0):
    # For each t, change token t + 1 and measure how far outputs at positions 0..t move.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, C, generator=g)
    worst = 0.0
    with torch.no_grad():
        y = attn(x)
        for t in range(T - 1):
            x2 = x.clone()
            x2[:, t + 1] += torch.randn(C, generator=g)
            worst = max(worst, (y[:, : t + 1] - attn(x2)[:, : t + 1]).abs().max().item())
    return worst

class NoMask(CausalSelfAttention):
    def attend(self, q, k, v):
        return torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d), dim=-1) @ v

class MaskAfterSoftmax(CausalSelfAttention):
    def attend(self, q, k, v):
        T = q.shape[-2]
        w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d), dim=-1)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return w.masked_fill(future, 0.0) @ v                   # zeroed after normalizing

torch.manual_seed(0)
base = CausalSelfAttention(C, h).eval()
for cls in [CausalSelfAttention, SDPAAttention, NoMask, MaskAfterSoftmax]:
    attn = cls(C, h).eval()
    attn.load_state_dict(base.state_dict())
    worst = leakage(attn)
    print(cls.__name__.ljust(20), "largest change at positions <= t:", f"{worst:.2e}",
          "PASS" if worst < 1e-6 else "LEAKS")
