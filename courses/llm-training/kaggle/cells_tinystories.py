CELLS = [
("md", "# LLM Training Lab: pretrain a GPT from random init on TinyStories (T4 test run)\n\nFROM SCRATCH. Our own GPT (pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings), our own BPE tokenizer, fp16 autocast + GradScaler on a T4. Logs val loss, bits per byte and samples from fixed prompts, and writes a Run Library JSON plus checkpoints."),
("code", r'''
import json, math, os, platform, subprocess, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SEED = int(os.environ.get("LLT_SEED", "0"))
CFG = dict(vocab=4096, d=320, layers=8, heads=8, ctx=512, batch=64, lr=1e-3, warmup=200,
           steps=int(os.environ.get("LLT_STEPS", "3000")), eval_every=250, eval_batches=40, wd=0.1, clip=1.0)
torch.manual_seed(SEED); np.random.seed(SEED)
assert torch.cuda.is_available(), "No GPU: Settings > Accelerator > GPU T4"
print(torch.cuda.get_device_name(0), "bf16" if torch.cuda.is_bf16_supported() else "no bf16 (use fp16 + GradScaler)")
RUN = {"runId": f"lltTinyStories-T2-baseline-s{SEED}", "module": "lltTinyStories", "tier": "T2", "label": "baseline",
       "hardware": {"gpu": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                    "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)},
       "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
       "config": CFG, "seed": SEED, "metrics": [], "samples": [], "date": time.strftime("%Y-%m-%d")}
T0 = time.time()
'''),
("code", r'''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets", "tokenizers"], check=False)
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
train_text, val_text = ds["train"]["text"], ds["validation"]["text"]
RUN["data"] = {"dataset": "roneneldan/TinyStories", "trainStories": len(train_text), "valStories": len(val_text)}
print(RUN["data"])
'''),
("code", r'''
# Byte-level BPE trained on the TRAIN split only (training on validation is a listed gotcha).
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=CFG["vocab"], special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
t = time.time(); tok.train_from_iterator(train_text[:300000], trainer=trainer)
EOT = tok.token_to_id("<|eot|>")
RUN["tokenizerTrainSec"] = round(time.time() - t, 1)

def encode_all(texts):
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids); ids.append(EOT)
    return np.array(ids, dtype=np.uint16)

t = time.time()
train_ids = np.concatenate([encode_all(train_text[i:i + 100000]) for i in range(0, len(train_text), 100000)])
val_ids = encode_all(val_text)
RUN["data"].update(trainTokens=int(train_ids.size), valTokens=int(val_ids.size), encodeSec=round(time.time() - t, 1),
                   bytesPerToken=round(sum(len(s.encode()) for s in val_text) / (val_ids.size - len(val_text)), 3))
print(RUN["data"])
'''),
("code", r'''
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)

def rope_tables(T, hd, device):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        hid = int(8 * d / 3 / 64 + 0.999) * 64
        self.w1 = nn.Linear(d, hid, bias=False); self.w3 = nn.Linear(d, hid, bias=False); self.w2 = nn.Linear(hid, d, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))

class GPT(nn.Module):
    def __init__(self, V, d, L, H, ctx):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.blocks = nn.ModuleList([Block(d, H) for _ in range(L)])
        self.norm = RMSNorm(d); self.head = nn.Linear(d, V, bias=False); self.head.weight = self.emb.weight
        self.H, self.ctx = H, ctx
        for n, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(("o.weight", "w2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
    def forward(self, idx):
        T = idx.shape[1]
        cos, sin = rope_tables(T, self.emb.embedding_dim // self.H, idx.device)
        x = self.emb(idx)
        for b in self.blocks: x = b(x, cos, sin)
        return self.head(self.norm(x))

model = GPT(CFG["vocab"], CFG["d"], CFG["layers"], CFG["heads"], CFG["ctx"]).cuda()
RUN["params"] = sum(p.numel() for p in model.parameters())
print("params", RUN["params"])
'''),
("code", r'''
def batch(arr, B, T):
    ix = np.random.randint(0, arr.size - T - 1, B)
    x = torch.from_numpy(np.stack([arr[i:i + T].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([arr[i + 1:i + 1 + T].astype(np.int64) for i in ix]))
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

@torch.no_grad()
def evaluate():
    model.eval(); nll, ntok, nbytes = 0.0, 0, 0
    T, n = CFG["ctx"], CFG["eval_batches"] * 16
    for j in range(0, min(n * T, val_ids.size - T - 1), T * 16):
        xs = [val_ids[s:s + T].astype(np.int64) for s in range(j, j + 16 * T, T) if s + T + 1 <= val_ids.size]
        x = torch.tensor(np.stack(xs)).cuda()
        y = torch.tensor(np.stack([val_ids[s + 1:s + T + 1].astype(np.int64) for s in range(j, j + 16 * T, T) if s + T + 1 <= val_ids.size])).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        nll += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
        ntok += y.numel()
        nbytes += sum(len(tok.decode([int(t) for t in row if t != EOT]).encode()) + int((row == EOT).sum()) for row in y.cpu().numpy())
    model.train()
    return nll / ntok, nll / (math.log(2) * nbytes)

@torch.no_grad()
def sample(prompt, n=80, temp=0.8, topk=50):
    model.eval(); g = torch.Generator(device="cuda").manual_seed(1234)
    x = torch.tensor([tok.encode(prompt).ids], device="cuda")
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x[:, -CFG["ctx"]:])[:, -1, :].float() / temp
        v, _ = torch.topk(lg, topk); lg[lg < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
        if nxt.item() == EOT: break
        x = torch.cat([x, nxt], 1)
    model.train()
    return tok.decode(x[0].tolist())

PROMPTS = ["Once upon a time", "Tom and his dog went to the park. Tom"]
print("step 0 val loss/bpb", evaluate(), "ln(V) =", round(math.log(CFG["vocab"]), 3))
'''),
("code", r'''
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=CFG["lr"], betas=(0.9, 0.95), fused=True)
scaler = torch.amp.GradScaler("cuda")
def lr_at(s):
    if s < CFG["warmup"]: return CFG["lr"] * (s + 1) / CFG["warmup"]
    p = (s - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"])
    return CFG["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

os.makedirs("/kaggle/working/ckpt", exist_ok=True)
torch.cuda.reset_peak_memory_stats(); t_train = time.time(); t_last = t_train; tok_since = 0
for step in range(CFG["steps"] + 1):
    if step % CFG["eval_every"] == 0 or step == CFG["steps"]:
        vl, bpb = evaluate()
        el = time.time() - t_train
        m = {"step": step, "tokens": step * CFG["batch"] * CFG["ctx"], "valLoss": round(vl, 4), "valBpb": round(bpb, 4),
             "lr": lr_at(step), "elapsedSec": round(el, 1), "memGB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        if step: m["tokPerSec"] = int(tok_since / (time.time() - t_last)); m["trainLoss"] = round(train_loss, 4)
        RUN["metrics"].append(m); print(m); t_last = time.time(); tok_since = 0
        if step % 1000 == 0 or step == CFG["steps"]:
            RUN["samples"] += [{"step": step, "prompt": p, "completion": sample(p)} for p in PROMPTS]
            print(RUN["samples"][-2:])
            torch.save({"model": model.state_dict(), "step": step, "cfg": CFG}, f"/kaggle/working/ckpt/step{step}.pt")
    if step == CFG["steps"]: break
    for gparam in opt.param_groups: gparam["lr"] = lr_at(step)
    x, y = batch(train_ids, CFG["batch"], CFG["ctx"])
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
    scaler.step(opt); scaler.update()
    train_loss = loss.item() if step % 50 == 0 else train_loss if step else loss.item()
    tok_since += CFG["batch"] * CFG["ctx"]
'''),
("code", r'''
tok.save("/kaggle/working/ckpt/tokenizer.json")
RUN["wallClockSec"] = round(time.time() - T0, 1)
RUN["trainSec"] = round(time.time() - t_train, 1)
RUN["peakMemGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
with open(f"/kaggle/working/{RUN['runId']}.json", "w") as f: json.dump(RUN, f, indent=1)
print(json.dumps({k: RUN[k] for k in ["runId", "params", "wallClockSec", "trainSec", "peakMemGB"]}, indent=1))
print(RUN["metrics"][-1]); print(RUN["samples"][-2:])
'''),
]
