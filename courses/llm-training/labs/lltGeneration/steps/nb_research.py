# Research extension (optional). Induction score and ICL score on a 2-layer attention-only model
# trained FROM SCRATCH on random tokens that repeat with a period of 8 to 32. Same setup as induction.py
# (run lltGeneration-t1-induction-toy-L2-s0). Set LAYERS = 1 for the control, where no induction head can form.
LAYERS, V, D, H, N, STEPS = 2, 128, 64, 4, 32, 1500

class AttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)

class AttnOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([AttnBlock() for _ in range(LAYERS)])
        self.norm = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * LAYERS) if n.endswith("o.weight") else 0.02)
    def forward(self, idx, return_maps=False):
        B, T = idx.shape
        cos, sin = rope_tables(0, T, D // H)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        x, maps = self.emb(idx), []
        for b in self.blocks:
            q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in b.qkv(b.n1(x)).split(D, 2))
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            att = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(~mask, float("-inf")).softmax(-1)
            maps.append(att)
            x = x + b.o((att @ v).transpose(1, 2).reshape(B, T, D))
        logits = self.head(self.norm(x))
        return (logits, maps) if return_maps else logits

def train_batch(batch, gen):
    # n random tokens repeated to length 64, n drawn from 8 to 32 for each sequence.
    # The period varies, so a head that looks a fixed distance back cannot predict the repeats.
    x = torch.empty(batch, 2 * N, dtype=torch.long)
    for b in range(batch):
        n = int(torch.randint(8, 33, (1,), generator=gen))
        x[b] = torch.randint(0, V, (n,), generator=gen).repeat(2 * N // n + 1)[:2 * N]
    return x

def probe(seed):
    r = torch.randint(0, V, (64, N), generator=torch.Generator().manual_seed(seed))
    return torch.cat([r, r], 1)                            # 32 random tokens, then the same 32 again

@torch.no_grad()
def scores(m):
    _, maps = m(probe(1234), return_maps=True)             # the same probes as induction.py
    dst = torch.arange(N, 2 * N)                           # every token of the second copy
    ind = maps[-1][:, :, dst, dst - N + 1].mean((0, 2))    # attention to the token after its first occurrence
    x = probe(999)
    logits = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(64, -1)
    icl = (loss[:, 59] - loss[:, 9]).mean().item()         # loss at token 60 minus loss at token 10
    return ind.max().item(), icl

torch.manual_seed(0)
toy = AttnOnly()
opt = torch.optim.AdamW(toy.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
gen = torch.Generator().manual_seed(100)
for step in range(STEPS + 1):
    if step % 50 == 0:
        toy.eval(); ind, icl = scores(toy); toy.train()
        print(f"step {step:4d}  induction score {ind:.2f}  ICL score {icl:+.2f} nats")
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = 1e-3 * min(1.0, (step + 1) / 100)
    x = train_batch(32, gen)
    loss = F.cross_entropy(toy(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
