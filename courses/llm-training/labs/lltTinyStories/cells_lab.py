"""Cells of the learner notebook lltTinyStories.ipynb (build with build_lab_nb.py).

The same cells run on a free T4 (T2, Colab or Kaggle) and on a laptop CPU (T1); the first code cell
picks the tier. run_cells.py executes them in order on this machine for the T1 exactness checks.
No backslashes in code: the lab page quotes these cells inside JS template literals.
"""
CELLS = [
("md", """# lltTinyStories: your first real pretraining run

LLM Training Lab, module 7. Regime FROM SCRATCH: random weights, every weight changes, objective next-token NLL.

Run the cells in order. On a T4 (Colab or Kaggle) this trains our 11.5M-parameter GPT on 98.3M TinyStories tokens. Without a GPU it trains a 3.0M model on a smaller budget (T1).

Set `FROM_OUR_CHECKPOINT = True` in the config cell to load our weights at step 2,000 and run only the last 1,000 steps."""),
("code", r'''
# Step 1. Runtime check. Run this first; it prints the fix for each problem it finds.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU; set before torch loads (Kaggle gives two T4s)
import sys, platform, shutil
problems = []
if sys.version_info < (3, 11):
    problems.append("Python " + platform.python_version() + " is too old. Use a current Colab or Kaggle image (3.11 or newer, 3.12 recommended).")
import torch
if torch.cuda.is_available():
    GPU = torch.cuda.get_device_name(0)
    CAP = torch.cuda.get_device_capability(0)
    if CAP < (7, 0):
        problems.append(GPU + " is not supported by current PyTorch wheels. Kaggle, Settings > Accelerator > GPU T4 x2. Colab, Runtime > Change runtime type > T4 GPU.")
    TIER = "T2"
else:
    GPU, CAP, TIER = None, (0, 0), "T1"
    print("No GPU. Running the CPU config (T1). For a T4, Colab, Runtime > Change runtime type > T4 GPU.")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
# Pick the dtype from the hardware. torch.cuda.is_bf16_supported() returns True on a T4,
# which has no bf16 tensor cores, so it is the wrong test.
AMP_DTYPE = torch.bfloat16 if CAP >= (8, 0) else torch.float16
free_gb = shutil.disk_usage(".").free / 1e9
if free_gb < 4:
    problems.append("Only " + str(round(free_gb, 1)) + " GB disk free; the dataset and token cache need about 4 GB.")
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    if ram_gb < 8:
        problems.append("Only " + str(round(ram_gb, 1)) + " GB RAM; encoding the full split needs about 8 GB.")
except ImportError:
    ram_gb = None
import numpy
print("python", platform.python_version(), "| torch", torch.__version__, "| GPU", GPU, "| capability", CAP)
print("numpy", numpy.__version__, "found (our T1 runs used 2.5.4; GPU images keep their own numpy)")
print("tier", TIER, "| autocast dtype", str(AMP_DTYPE), "| disk free GB", round(free_gb, 1), "| RAM GB", ram_gb and round(ram_gb, 1))
for p in problems:
    print("PROBLEM", p)
if not problems:
    print("Runtime OK")
'''),
("code", r'''
# Step 1b. Pin the library versions this lab was measured with (T1 run, 2026-11-05).
# If the cell installs anything, restart the session once and run from step 1 again.
import importlib.metadata as md, subprocess
PINS = {"datasets": "5.1.0", "tokenizers": "0.23.3"}   # numpy is not pinned: the image's torch is built against its own numpy
def installed(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None
todo = [p + "==" + v for p, v in PINS.items() if installed(p) != v]
if todo:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + todo, check=True)
    raise SystemExit("Installed " + ", ".join(todo) + ". Restart the session, then run from step 1.")
print("versions OK", PINS)
'''),
("code", r'''
# Step 2. Config. Change one value at a time for the experiments.
FROM_OUR_CHECKPOINT = False   # True loads our step-2,000 weights (T2 only) and trains steps 2,000 to 3,000
SEED = 0
CONFIGS = {
    "T2": dict(vocab=4096, d=320, layers=8, heads=8, ctx=512, batch=64, lr=1e-3, warmup=200, steps=3000,
               eval_every=250, eval_batches=40, wd=0.1, clip=1.0, train_stories=None),
    "T1": dict(vocab=4096, d=192, layers=5, heads=6, ctx=256, batch=32, lr=1e-3, warmup=100, steps=1500,
               eval_every=250, eval_batches=10, wd=0.1, clip=1.0, train_stories=200000),
}
CFG = dict(CONFIGS[TIER])
CFG["steps"] = int(os.environ.get("LLT_STEPS", CFG["steps"]))
OUR_FILES = "https://huggingface.co/aimlcompanion/llm-training-lab/resolve/main/lltTinyStories/"   # our files on the Hugging Face Hub
CACHE = os.environ.get("LLT_CACHE", "llt_cache")
os.makedirs(CACHE, exist_ok=True)
if FROM_OUR_CHECKPOINT and TIER != "T2":
    raise SystemExit("Our checkpoint is the 11.5M T2 model. Without a GPU, use the recorded runs on the lab page.")
print(TIER, CFG)
'''),
("code", r'''
# Step 3. Tokenizer on the TRAIN split only, then encode once and cache the ids as uint16 files.
import json, time, urllib.request
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tag = TIER + ("-ours" if FROM_OUR_CHECKPOINT else "")
paths = {k: os.path.join(CACHE, tag + "-" + k) for k in ("tokenizer.json", "train.npy", "val.npy")}
t0 = time.time()
if all(os.path.exists(p) for p in paths.values()):
    tok = Tokenizer.from_file(paths["tokenizer.json"])
    train_ids = np.load(paths["train.npy"], mmap_mode="r")
    val_ids = np.load(paths["val.npy"])
    print("loaded the cached token ids in", round(time.time() - t0, 1), "s")
else:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    val_text = ds["validation"]["text"]
    n_train = 400000 if FROM_OUR_CHECKPOINT else CFG["train_stories"]
    train_text = ds["train"].select(range(n_train))["text"] if n_train else ds["train"]["text"]
    if FROM_OUR_CHECKPOINT:
        urllib.request.urlretrieve(OUR_FILES + "tokenizer.json", paths["tokenizer.json"])
        tok = Tokenizer.from_file(paths["tokenizer.json"])
    else:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=CFG["vocab"], special_tokens=["<|eot|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        t = time.time()
        tok.train_from_iterator(ds["train"].select(range(300000))["text"], trainer=trainer)   # same 300k stories at every tier
        print("tokenizer trained in", round(time.time() - t, 1), "s")
        tok.save(paths["tokenizer.json"])
    eot = tok.token_to_id("<|eot|>")
    enc = getattr(tok, "encode_batch_fast", None) or tok.encode_batch
    def encode_all(texts):
        parts = []
        for e in enc(texts):
            parts.append(np.asarray(e.ids, dtype=np.uint16))
            parts.append(np.array([eot], dtype=np.uint16))
        return np.concatenate(parts)
    t = time.time()
    train_ids = np.concatenate([encode_all(train_text[i:i + 100000]) for i in range(0, len(train_text), 100000)])
    val_ids = encode_all(val_text)
    print("encoded", len(train_text), "train stories in", round(time.time() - t, 1), "s")
    np.save(paths["train.npy"], train_ids)
    np.save(paths["val.npy"], val_ids)
EOT = tok.token_to_id("<|eot|>")
print("train tokens", train_ids.size, "| val tokens", val_ids.size)
'''),
("code", r'''
# Step 3b. Exact bytes per token id, for bits per byte. Byte-level BPE writes each byte as one
# printable character (the GPT-2 table), so a token's length in characters is its length in bytes.
def byte_level_alphabet():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    extra = [b for b in range(256) if b not in bs]
    return set(chr(b) for b in bs) | set(chr(256 + i) for i in range(len(extra)))
ALPHABET = byte_level_alphabet()
TOKEN_BYTES = np.zeros(tok.get_vocab_size(), dtype=np.int64)
for i in range(tok.get_vocab_size()):
    s = tok.id_to_token(i)
    TOKEN_BYTES[i] = 1 if i == EOT else len(s)   # the end-of-story token stands for one separator byte
    assert i == EOT or set(s) <= ALPHABET, s
print("bytes per token on val", round(TOKEN_BYTES[val_ids].sum() / val_ids.size, 3))
'''),
("code", r'''
# Step 4. The model: pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings (built in lltGPT).
import math
import torch.nn as nn
import torch.nn.functional as F

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
        cos, sin = rope_tables(idx.shape[1], self.emb.embedding_dim // self.H, idx.device)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))

DEV = "cuda" if TIER == "T2" else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
model = GPT(CFG["vocab"], CFG["d"], CFG["layers"], CFG["heads"], CFG["ctx"]).to(DEV)
N_PARAMS = sum(p.numel() for p in model.parameters())
print("parameters", N_PARAMS, "| tokens per step", CFG["batch"] * CFG["ctx"],
      "| tokens per parameter", round(CFG["steps"] * CFG["batch"] * CFG["ctx"] / N_PARAMS, 1))
'''),
("code", r'''
# Step 5. Validation loss, bits per byte, and samples from fixed prompts.
def autocast():
    return torch.autocast(DEV, dtype=AMP_DTYPE, enabled=DEV == "cuda")

@torch.no_grad()
def evaluate():
    model.eval()
    nll, ntok, nbytes = 0.0, 0, 0
    T = CFG["ctx"]
    for j in range(0, CFG["eval_batches"] * 16 * T, 16 * T):
        starts = [s for s in range(j, j + 16 * T, T) if s + T + 1 <= val_ids.size]
        x = torch.tensor(np.stack([val_ids[s:s + T].astype(np.int64) for s in starts]), device=DEV)
        y_np = np.stack([val_ids[s + 1:s + T + 1].astype(np.int64) for s in starts])
        with autocast():
            logits = model(x)
        nll += F.cross_entropy(logits.float().view(-1, logits.size(-1)), torch.tensor(y_np, device=DEV).view(-1), reduction="sum").item()
        ntok += y_np.size
        nbytes += int(TOKEN_BYTES[y_np].sum())
    model.train()
    return nll / ntok, nll / (math.log(2) * nbytes)

@torch.no_grad()
def sample(prompt, n=80, temp=0.8, topk=50):
    model.eval()
    g = torch.Generator(device=DEV).manual_seed(1234)
    x = torch.tensor([tok.encode(prompt).ids], device=DEV)
    for _ in range(n):
        with autocast():
            lg = model(x[:, -CFG["ctx"]:])[:, -1, :].float() / temp
        v, _ = torch.topk(lg, topk)
        lg[lg < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
        if nxt.item() == EOT:
            break
        x = torch.cat([x, nxt], 1)
    model.train()
    return tok.decode(x[0].tolist())

PROMPTS = ["Once upon a time", "Tom and his dog went to the park. Tom"]
vl, bpb = evaluate()
print("step 0 val loss", round(vl, 4), "| ln(V)", round(math.log(CFG["vocab"]), 4), "| bits per byte", round(bpb, 4))
'''),
("code", r'''
# Step 6. Train. Checkpoints every 20 minutes; re-running this cell resumes exactly.
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=CFG["lr"], betas=(0.9, 0.95), fused=DEV == "cuda")
scaler = torch.amp.GradScaler("cuda") if AMP_DTYPE == torch.float16 and DEV == "cuda" else None

def lr_at(s):
    if s < CFG["warmup"]:
        return CFG["lr"] * (s + 1) / CFG["warmup"]
    p = (s - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"])
    return CFG["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

def get_batch():
    ix = np.random.randint(0, train_ids.size - CFG["ctx"] - 1, CFG["batch"])
    x = np.stack([np.asarray(train_ids[i:i + CFG["ctx"]], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(train_ids[i + 1:i + 1 + CFG["ctx"]], dtype=np.int64) for i in ix])
    return torch.from_numpy(x).to(DEV), torch.from_numpy(y).to(DEV)

CKPT = os.path.join(CACHE, "ckpt-" + TIER + "-s" + str(SEED) + ".pt")
start, REWARM = 0, 0
LOG = {"metrics": [], "samples": []}
if FROM_OUR_CHECKPOINT:
    urllib.request.urlretrieve(OUR_FILES + "step2000.pt", os.path.join(CACHE, "our-step2000.pt"))
    model.load_state_dict(torch.load(os.path.join(CACHE, "our-step2000.pt"), map_location=DEV, weights_only=True)["model"])
    start, REWARM = 2000, 0   # weights only, fresh AdamW state; no re-warmup needed (run lltTinyStories-T2-baseline-s1-resume2000-rewarm0)
elif os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
    start, LOG = ck["step"], ck["log"]
    print("resumed at step", start)

t_save = time.time(); t_train = time.time()
for step in range(start, CFG["steps"] + 1):
    if step % CFG["eval_every"] == 0 or step == CFG["steps"]:
        vl, bpb = evaluate()
        m = {"step": step, "valLoss": round(vl, 4), "valBpb": round(bpb, 4), "lr": lr_at(step), "sec": round(time.time() - t_train, 1)}
        LOG["metrics"].append(m)
        print(m)
        if step % 1000 == 0 or step == CFG["steps"]:
            for p in PROMPTS:
                LOG["samples"].append({"step": step, "prompt": p, "completion": sample(p)})
                print(step, repr(LOG["samples"][-1]["completion"][:300]))
    if step == CFG["steps"]:
        break
    lr = lr_at(step) * (min(1.0, (step - start + 1) / REWARM) if REWARM else 1.0)
    for group in opt.param_groups:
        group["lr"] = lr
    x, y = get_batch()
    with autocast():
        logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad(set_to_none=True)
    if scaler:
        scaler.scale(loss).backward(); scaler.unscale_(opt)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
    if scaler:
        scaler.step(opt); scaler.update()
    else:
        opt.step()
    if time.time() - t_save > 20 * 60:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "step": step + 1, "log": LOG}, CKPT)
        t_save = time.time()
TRAIN_SEC = time.time() - t_train
print("trained in", round(TRAIN_SEC), "s")
'''),
("code", r'''
# Step 7. Compare with our runs and save your run file.
BANDS = {"T2": (1.62, 1.67), "T1": (2.05, 2.12)}   # final val loss: our T2 seeds 1.6445 and 1.6433, our T1 run 2.0833
final = LOG["metrics"][-1]
lo, hi = BANDS[TIER]
if CFG["steps"] == CONFIGS[TIER]["steps"] and lo is not None:
    print("final val loss", final["valLoss"], "| our band", (lo, hi), "| inside" if lo <= final["valLoss"] <= hi else "| OUTSIDE: check the config cell")
run = {"runId": "my-lltTinyStories-" + TIER + "-s" + str(SEED), "module": "lltTinyStories", "tier": TIER,
       "env": {"python": platform.python_version(), "torch": torch.__version__, "gpu": GPU},
       "config": CFG, "seed": SEED, "params": N_PARAMS, "fromOurCheckpoint": FROM_OUR_CHECKPOINT,
       "trainSec": round(TRAIN_SEC, 1), **LOG}
with open(run["runId"] + ".json", "w") as f:
    json.dump(run, f, indent=1)
print("wrote", run["runId"] + ".json")
'''),
]
