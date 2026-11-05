import math, random, time
import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Pretrain the lab model FROM SCRATCH on CPU (same recipe as train_small.py, run lltGeneration-t1-model-s0).
torch.set_num_threads(4)
path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
stories = [s.strip() for s in open(path, encoding="utf-8").read().split("<|endoftext|>") if s.strip()]
random.Random(0).shuffle(stories)
cut = int(0.95 * len(stories))
train_text, val_text = stories[:cut], stories[cut:]

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(train_text, trainers.BpeTrainer(vocab_size=2048, special_tokens=["<|eot|>"],
                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
EOT = tok.token_to_id("<|eot|>")
def encode(texts):
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(EOT)
    return np.array(ids, dtype=np.int64)
tr, va = encode(train_text), encode(val_text)

torch.manual_seed(0)
rng = np.random.default_rng(0)
model = GPT(2048, 192, 4, 6, 256)
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=2e-3, betas=(0.9, 0.95))
STEPS, T, B = 1500, 256, 16
vx = torch.tensor(np.stack([va[s:s + T] for s in range(0, 64 * T, T)]))
vy = torch.tensor(np.stack([va[s + 1:s + T + 1] for s in range(0, 64 * T, T)]))
def lr_at(s):
    if s < 100:
        return 2e-3 * (s + 1) / 100
    return 2e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 100) / (STEPS - 100))))
t0 = time.time()
for step in range(STEPS + 1):
    if step % 250 == 0:
        with torch.no_grad():
            model.eval()
            vl = sum(F.cross_entropy(model(vx[i:i + 16]).view(-1, 2048), vy[i:i + 16].reshape(-1), reduction="sum").item()
                     for i in range(0, 64, 16)) / vy.numel()
            model.train()
        print(f"step {step:5d}  val loss {vl:.4f}  {time.time() - t0:.0f} s")
    if step == STEPS:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    ix = rng.integers(0, tr.size - T - 1, B)
    x = torch.tensor(np.stack([tr[i:i + T] for i in ix]))
    y = torch.tensor(np.stack([tr[i + 1:i + T + 1] for i in ix]))
    loss = F.cross_entropy(model(x).view(-1, 2048), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
model.eval()
print("parameters", sum(p.numel() for p in model.parameters()))
