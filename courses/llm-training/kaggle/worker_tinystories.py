"""Training worker for the lltTinyStories sweep (kernel llt-tinystories-sweep).

Runs a queue of configs on ONE GPU (the launcher sets CUDA_VISIBLE_DEVICES before this process
imports torch). Reads token ids from a shared .npy cache, so the corpus is encoded once per
session, not once per run. Writes one Run Library JSON per run as soon as that run ends.

Usage: python worker_tinystories.py <queue.json> <cache_dir> <out_dir>
"""
import json, math, os, platform, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

queue_path, cache_dir, out_dir = sys.argv[1:4]
QUEUE = json.load(open(queue_path))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu": torch.set_num_threads(4)
CAP = torch.cuda.get_device_capability(0) if DEV == "cuda" else (0, 0)
# Pick the dtype from the hardware, not from torch.cuda.is_bf16_supported(): that call returns True
# on a T4, which has no bf16 tensor cores (measured, run llt-gpu-smoke).
AMP_DTYPE = torch.bfloat16 if CAP >= (8, 0) else torch.float16
USE_SCALER = DEV == "cuda" and AMP_DTYPE == torch.float16

train_ids = np.load(os.path.join(cache_dir, "train_ids.npy"), mmap_mode="r")
val_ids = np.load(os.path.join(cache_dir, "val_ids.npy"))
tok_bytes = np.load(os.path.join(cache_dir, "token_bytes.npy"))  # bytes per token id (EOT counts 1)
meta = json.load(open(os.path.join(cache_dir, "meta.json")))
VAL_B_START = meta["valBStartToken"]
EOT = meta["eot"]

from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(cache_dir, "tokenizer.json"))


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


def autocast():
    return torch.autocast(DEV, dtype=AMP_DTYPE, enabled=DEV == "cuda")


def run_one(C):
    seed = C["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    ctx, B, micro = C["ctx"], C["batch"], C.get("micro", C["batch"])
    accum = B // micro
    model = GPT(C["vocab"], C["d"], C["layers"], C["heads"], ctx).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.emb.weight.numel()
    RUN = {"runId": C["runId"], "module": "lltTinyStories", "tier": C.get("tier", "T2"), "label": C["label"],
           "hardware": {"gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
                        "count": torch.cuda.device_count() if DEV == "cuda" else 0,
                        "vramGB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if DEV == "cuda" else None,
                        "capability": list(CAP), "concurrent": C.get("concurrent", "one run per GPU, two GPUs busy")},
           "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "ampDtype": str(AMP_DTYPE).replace("torch.", ""), "gradScaler": USE_SCALER},
           "config": {k: v for k, v in C.items() if k not in ("runId", "label", "tier")},
           "seed": seed, "params": n_params, "nonEmbeddingParams": n_params - n_emb,
           "flopsPerToken6N": 6 * n_params, "metrics": [], "trainCurve": [], "samples": [],
           "data": meta["data"], "date": time.strftime("%Y-%m-%d")}

    def batch():
        ix = np.random.randint(0, train_ids.size - ctx - 1, micro)
        x = torch.from_numpy(np.stack([np.asarray(train_ids[i:i + ctx], dtype=np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([np.asarray(train_ids[i + 1:i + 1 + ctx], dtype=np.int64) for i in ix]))
        return x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)

    @torch.no_grad()
    def evaluate(start):
        # Same window as the baseline: 640 sequences of ctx tokens from `start`. Bits per byte use
        # the exact byte length of every target token.
        model.eval(); nll, ntok, nbytes = 0.0, 0, 0
        n = C["eval_batches"] * 16
        for j in range(start, start + n * ctx, ctx * 16):
            starts = [s for s in range(j, j + 16 * ctx, ctx) if s + ctx + 1 <= val_ids.size]
            if not starts: break
            x = torch.tensor(np.stack([val_ids[s:s + ctx].astype(np.int64) for s in starts]), device=DEV)
            y_np = np.stack([val_ids[s + 1:s + ctx + 1].astype(np.int64) for s in starts])
            y = torch.tensor(y_np, device=DEV)
            with autocast():
                logits = model(x)
            nll += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()
            ntok += y.numel(); nbytes += int(tok_bytes[y_np].sum())
        model.train()
        return nll / ntok, nll / (math.log(2) * nbytes), ntok, nbytes

    @torch.no_grad()
    def sample(prompt, n=80, temp=0.8, topk=50):
        model.eval(); g = torch.Generator(device=DEV).manual_seed(1234)
        x = torch.tensor([tok.encode(prompt).ids], device=DEV)
        for _ in range(n):
            with autocast():
                lg = model(x[:, -ctx:])[:, -1, :].float() / temp
            v, _ = torch.topk(lg, topk); lg[lg < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
            if nxt.item() == EOT: break
            x = torch.cat([x, nxt], 1)
        model.train()
        return tok.decode(x[0].tolist())

    def lr_at(s, total, peak, warmup, offset=0, rewarm=0):
        if s < warmup: lr = peak * (s + 1) / warmup
        else:
            p = (s - warmup) / max(1, total - warmup)
            lr = peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        if rewarm and s - offset < rewarm: lr = lr * (s - offset + 1) / rewarm
        return lr

    def make_opt():
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": C["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                                 lr=C["lr"], betas=(0.9, 0.95), fused=DEV == "cuda")

    def safe(v):
        return None if (v is None or not math.isfinite(v)) else round(v, 4)

    def train(start_step, end_step, opt, scaler, record, rewarm=0, save_at=None):
        t_last = time.time(); tok_since = 0; run_loss = 0.0; run_n = 0; run_gn = 0.0; bad = 0
        t0 = time.time()
        for step in range(start_step, end_step + 1):
            if record and (step % C["eval_every"] == 0 or step == end_step):
                vl, bpb, ntok, nbytes = evaluate(0)
                vlb, bpbb, _, _ = evaluate(VAL_B_START)
                m = {"step": step, "tokens": step * B * ctx, "valLoss": safe(vl), "valBpb": safe(bpb),
                     "valBLoss": safe(vlb), "valBBpb": safe(bpbb), "lr": lr_at(step, C["steps"], C["lr"], C["warmup"], start_step, rewarm),
                     "elapsedSec": round(time.time() - t0, 1)}
                if DEV == "cuda": m["memGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if step > start_step: m["tokPerSec"] = int(tok_since / (time.time() - t_last))
                record["metrics"].append(m); print(C["runId"], m, flush=True)
                record.setdefault("evalTokens", ntok); record.setdefault("evalBytes", nbytes)
                t_last = time.time(); tok_since = 0
                if not math.isfinite(vl) or (step >= 1000 and vl > 6.0):
                    bad += 1
                    if bad >= 2 and C.get("stopIfDiverged"):
                        record["stoppedEarly"] = step; break
                if step == end_step or step % 1000 == 0:
                    record["samples"] += [{"step": step, "prompt": p, "completion": sample(p)} for p in C["prompts"]]
            if save_at is not None and step == save_at:
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, os.path.join("/tmp", C["runId"] + f"-step{step}.pt"))
            if step == end_step: break
            for gparam in opt.param_groups: gparam["lr"] = lr_at(step, C["steps"], C["lr"], C["warmup"], start_step, rewarm)
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(accum):
                x, y = batch()
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)) / accum
                (scaler.scale(loss) if scaler else loss).backward(); tot += loss.item()
            if scaler: scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C["clip"]).item()
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()
            run_loss += tot; run_n += 1; run_gn += gn if math.isfinite(gn) else 0.0
            if record and run_n == 25:
                record["trainCurve"].append({"step": step + 1, "trainLoss": safe(run_loss / 25), "gradNorm": safe(run_gn / 25)})
                run_loss = 0.0; run_n = 0; run_gn = 0.0
            tok_since += B * ctx
        return time.time() - t0

    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
    opt = make_opt(); scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None
    RUN["trainSec"] = round(train(0, C["steps"], opt, scaler, RUN, save_at=C.get("saveAt")), 1)
    if DEV == "cuda": RUN["peakMemGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if C.get("saveFinal"):
        torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "cfg": C},
                   os.path.join(out_dir, C["runId"] + "-final-fp16.pt"))
    with open(os.path.join(out_dir, C["runId"] + ".json"), "w") as f: json.dump(RUN, f, indent=1)
    print("WROTE", C["runId"], RUN["metrics"][-1], flush=True)

    # Resume tests for the "start from our checkpoint" exit: load model weights only (no optimizer
    # state, which is what our published baseline checkpoints hold) and finish the last stretch.
    for rw in C.get("resumeTests", []):
        sd = torch.load(os.path.join("/tmp", C["runId"] + f"-step{C['saveAt']}.pt"))
        model.load_state_dict(sd)
        torch.manual_seed(seed + 1000); np.random.seed(seed + 1000)
        R = {"runId": C["runId"] + f"-resume{C['saveAt']}-rewarm{rw}", "module": "lltTinyStories", "tier": "T2",
             "label": f"ablation:resume=weights-only,rewarm={rw}", "hardware": RUN["hardware"], "env": RUN["env"],
             "config": dict(RUN["config"], resumeFrom=C["saveAt"], rewarm=rw), "seed": seed, "params": n_params,
             "metrics": [], "trainCurve": [], "samples": [], "date": time.strftime("%Y-%m-%d")}
        opt = make_opt(); scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None
        R["trainSec"] = round(train(C["saveAt"], C["steps"], opt, scaler, R, rewarm=rw), 1)
        with open(os.path.join(out_dir, R["runId"] + ".json"), "w") as f: json.dump(R, f, indent=1)
        print("WROTE", R["runId"], R["metrics"][-1], flush=True)
    del model, opt
    if DEV == "cuda": torch.cuda.empty_cache()


for C in QUEUE:
    try:
        run_one(C)
    except Exception as e:  # one failed config must not cost the rest of the queue
        import traceback; traceback.print_exc()
        with open(os.path.join(out_dir, C["runId"] + ".error.txt"), "w") as f: f.write(repr(e))
        if DEV == "cuda": torch.cuda.empty_cache()
print("QUEUE DONE", flush=True)
