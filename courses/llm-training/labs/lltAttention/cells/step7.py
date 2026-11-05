# Train the same small character model with three attention classes from steps 4 and 5.
import urllib.request

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
text = urllib.request.urlopen(URL).read().decode("utf-8")[:1_000_000]
chars = sorted(set(text))
data = torch.tensor([chars.index(c) for c in text])
train, val = data[:900_000], data[900_000:]

class Block(nn.Module):
    def __init__(self, attn_cls, C, h):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(C), nn.RMSNorm(C)
        self.attn = attn_cls(C, h)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class TinyLM(nn.Module):
    def __init__(self, attn_cls, V, C=64, h=4, L=4, ctx=128):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, C), nn.Embedding(ctx, C)
        self.blocks = nn.Sequential(*[Block(attn_cls, C, h) for _ in range(L)])
        self.head = nn.Linear(C, V)

    def forward(self, idx):
        return self.head(self.blocks(self.tok(idx) + self.pos(torch.arange(idx.shape[1]))))

def batch(split, B=32, T=128, g=None):
    i = torch.randint(len(split) - T - 1, (B,), generator=g)
    return torch.stack([split[j:j + T] for j in i]), torch.stack([split[j + 1:j + T + 1] for j in i])

@torch.no_grad()
def prefix_loss(model, x, y):     # score position t from tokens 0..t only, as generation does
    return torch.stack([F.cross_entropy(model(x[:, :t + 1])[:, -1], y[:, t]) for t in range(x.shape[1])]).mean().item()

xv, yv = batch(val, B=64, g=torch.Generator().manual_seed(1))
results, curves, models = {}, {}, {}
for cls in [CausalSelfAttention, NoMask, MaskAfterSoftmax]:
    torch.manual_seed(0)
    model = TinyLM(cls, len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    curve = []
    for step in range(600):
        x, y = batch(train, g=g)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(loss.item())
    model.eval()
    with torch.no_grad():
        full = F.cross_entropy(model(xv).flatten(0, 1), yv.flatten()).item()
    results[cls.__name__] = (loss.item(), full, prefix_loss(model, xv, yv))
    curves[cls.__name__], models[cls.__name__] = curve, model
    print(cls.__name__.ljust(20), "train %.3f | val, full sequence %.3f | val, prefix only %.3f" % results[cls.__name__])
