from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
stories = [s.strip() for s in open(path, encoding="utf-8").read().split("<|endoftext|>") if s.strip()]
split = int(len(stories) * 0.95)   # 95% to train on, the last 5% held out
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
bpe.train_from_iterator(stories[:split], trainers.BpeTrainer(vocab_size=TOGGLE.vocab, special_tokens=["<|eot|>"],
                                                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))

def encode(texts):
    ids = []
    for enc in bpe.encode_batch(texts):
        ids += enc.ids + [bpe.token_to_id("<|eot|>")]
    return torch.tensor(ids)

train_ids, val_ids = encode(stories[:split]), encode(stories[split:])
VAL_X = val_ids[: 256 * 128].view(256, 128)          # 256 fixed windows of 128 tokens
VAL_Y = val_ids[1 : 256 * 128 + 1].view(256, 128)
print(f"{len(stories):,} stories  train tokens {len(train_ids):,}  val tokens {len(val_ids):,}")

@torch.no_grad()
def val_loss(model):
    model.eval()
    loss = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in zip(VAL_X.split(64), VAL_Y.split(64))) / 4
    model.train()
    return loss

def train(cfg, steps=600, seed=0, lr=3e-3, batch=32, warmup=50, every=100):
    torch.manual_seed(seed)
    model, gen = GPT(cfg), torch.Generator().manual_seed(seed)
    groups = [{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
              {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    curve = [(0, val_loss(model))]
    for step in range(steps):
        frac = (step + 1) / warmup if step < warmup else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr * frac
        starts = torch.randint(0, len(train_ids) - cfg.ctx - 1, (batch,), generator=gen)
        x = torch.stack([train_ids[s : s + cfg.ctx] for s in starts])
        y = torch.stack([train_ids[s + 1 : s + cfg.ctx + 1] for s in starts])
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % every == 0:
            curve.append((step + 1, val_loss(model)))
    return model, curve
