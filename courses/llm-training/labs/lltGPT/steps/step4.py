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
torch.manual_seed(0)
model = GPT(TINY)
print("head is the embedding tensor:", model.head.weight is model.tok.weight)
print(f"init std  qkv {model.blocks[0].attn.qkv.weight.std():.4f}  proj {model.blocks[0].attn.proj.weight.std():.4f}"
      f"  down {model.blocks[0].mlp.down.weight.std():.4f}  target 0.02/sqrt(16) = {0.02 / math.sqrt(16):.4f}")
idx, targets = torch.randint(0, TINY.vocab, (2, 4, 128))
with torch.no_grad():
    loss = F.cross_entropy(model(idx).flatten(0, 1), targets.flatten())
print(f"step-0 loss {loss:.3f}  ln(V) = {math.log(TINY.vocab):.3f}")
