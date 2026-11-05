"""lltPretrainData training worker (Kaggle GPU kernel llt-ptd-train).

Runs a queue of configs on ONE GPU (the launcher sets CUDA_VISIBLE_DEVICES before this process
imports torch). The model is the lltTinyStories GPT (kaggle/worker_tinystories.py): pre-norm,
RMSNorm, RoPE, SwiGLU, tied embeddings, fp16 autocast with GradScaler on a T4.

Unlike the TinyStories worker, every run here reads a FIXED token stream in a fixed shuffled
window order, so the number of times each document (and each canary copy) is seen is exact.

Usage: python worker_ptd.py <queue.json> <data_dir> <out_dir>
"""
import json, math, os, platform, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

queue_path, data_dir, out_dir = sys.argv[1:4]
QUEUE = json.load(open(queue_path))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    torch.set_num_threads(2)
CAP = torch.cuda.get_device_capability(0) if DEV == "cuda" else (0, 0)
# Pick the dtype from the hardware, not from torch.cuda.is_bf16_supported(), which returns True on a T4.
AMP_DTYPE = torch.bfloat16 if CAP >= (8, 0) else torch.float16
USE_SCALER = DEV == "cuda" and AMP_DTYPE == torch.float16
STATS = json.load(open(os.path.join(data_dir, "ptd_prep_stats.json")))
EOT = STATS["meta"]["eot"]
TOK_BYTES = np.load(os.path.join(data_dir, "token_bytes.npy"))
_CACHE = {}


def corpus(name):
    if name not in _CACHE:
        _CACHE[name] = (np.load(os.path.join(data_dir, name + ".npy"), mmap_mode="r"),
                        np.load(os.path.join(data_dir, name + "_offsets.npy")))
    return _CACHE[name]


def docs_of(name, max_tokens=None, max_docs=None):
    arr, off = corpus(name)
    ends = np.append(off[1:], arr.size)
    out, tot = [], 0
    for s, e in zip(off, ends):
        if max_docs is not None and len(out) >= max_docs:
            break
        if max_tokens is not None and tot >= max_tokens:
            break
        out.append(np.asarray(arr[s:e])); tot += e - s
    return out


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


def windows_from(stream, ctx):
    n = (stream.size - 1) // ctx
    return [stream[i * ctx:i * ctx + ctx + 1] for i in range(n)]


def build_stream(C, need_tokens, rng, record):
    """Return (list of windows of ctx+1 tokens in training order, anneal windows or None)."""
    ctx = C["ctx"]
    kind = C["data"]["kind"]
    if kind == "single":
        arr, _ = corpus(C["data"]["corpus"])
        unique = int(C["data"].get("uniqueTokens", need_tokens + 1))
        assert arr.size >= unique, (C["data"]["corpus"], arr.size, unique)
        base = np.asarray(arr[:unique])
        wins = windows_from(base, ctx)
        record["data"] = {"corpus": C["data"]["corpus"], "uniqueTokens": int(base.size), "windowsPerEpoch": len(wins)}
        order, epochs = [], 0
        while len(order) * 1 < need_tokens // ctx:
            p = rng.permutation(len(wins)); order.extend(p.tolist()); epochs += 1
        order = order[: need_tokens // ctx]
        record["data"]["epochs"] = round(len(order) / len(wins), 3)
        return [wins[i] for i in order]
    if kind == "canary":
        can = np.load(os.path.join(data_dir, "canaries.npy"))
        levels, per = C["data"]["levels"], C["data"]["perLevel"]
        docs = []
        plan = []
        k = 0
        for lvl in levels:
            for _ in range(per):
                plan.append((k, lvl)); k += 1
        assert k <= len(can), (k, len(can))
        for idx, lvl in plan:
            for _ in range(lvl):
                docs.append(("c", idx))
        n_can_tok = sum(lvl * 101 for _, lvl in plan)
        base = docs_of(C["data"]["corpus"], max_tokens=need_tokens + 1 - n_can_tok + 5000)
        docs += [("d", i) for i in range(len(base))]
        order = rng.permutation(len(docs))
        parts, counts = [], np.zeros(k, dtype=np.int64)
        tot = 0
        for j in order:
            t, i = docs[j]
            piece = np.append(can[i], np.uint16(EOT)) if t == "c" else base[i]
            parts.append(piece); tot += piece.size
        stream = np.concatenate(parts)[: need_tokens + 1]
        # Count the canary copies that survive the cut whole (the tail of the shuffled stream is cut).
        pos = 0
        for j in order:
            t, i = docs[j]
            L = 101 if t == "c" else base[i].size
            if pos + L <= stream.size and t == "c":
                counts[i] += 1
            pos += L
        record["canaryPlan"] = [{"index": int(i), "planned": int(l), "seen": int(counts[i])} for i, l in plan]
        record["data"] = {"corpus": C["data"]["corpus"], "canaryTokensPlanned": int(n_can_tok), "streamTokens": int(stream.size)}
        wins = windows_from(stream, ctx)
        return [wins[i] for i in rng.permutation(len(wins))]
    if kind == "mixture":
        total = need_tokens + 1
        doms = C["data"]["domains"]  # {name: weight}
        by = C["data"]["by"]
        pieces = {}
        if by == "tokens":
            for name, w in doms.items():
                pieces[name] = np.concatenate(docs_of(name, max_tokens=int(w * total) + 4 * ctx))[: int(w * total) + 2 * ctx]
        else:
            means = {}
            for name in doms:
                ds = docs_of(name, max_docs=4000)
                means[name] = float(np.mean([d.size for d in ds]))
            n_docs = 1.02 * total / sum(w * means[n] for n, w in doms.items())
            for name, w in doms.items():
                pieces[name] = np.concatenate(docs_of(name, max_docs=max(1, int(round(w * n_docs)))))
        record["data"] = {"by": by, "weights": doms,
                          "tokensPerDomain": {n: int(p.size) for n, p in pieces.items()},
                          "tokenShare": {n: round(p.size / sum(q.size for q in pieces.values()), 4) for n, p in pieces.items()}}
        wins = []
        for p in pieces.values():
            wins += windows_from(p, ctx)
        assert len(wins) >= need_tokens // ctx, ("mixture too short", len(wins), need_tokens // ctx)
        order = rng.permutation(len(wins))[: need_tokens // ctx]
        return [wins[i] for i in order]
    raise ValueError(kind)


def run_one(C):
    seed = C["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ctx, B, micro = C["ctx"], C["batch"], C.get("micro", C["batch"])
    accum = B // micro
    steps = C["steps"]
    model = GPT(C["vocab"], C["d"], C["layers"], C["heads"], ctx).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    R = {"runId": C["runId"], "module": "lltPretrainData", "tier": "T2", "label": C["label"],
         "hardware": {"gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None,
                      "count": torch.cuda.device_count() if DEV == "cuda" else 0,
                      "capability": list(CAP), "concurrent": "one run per GPU, two GPUs busy"},
         "env": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                 "ampDtype": str(AMP_DTYPE).replace("torch.", ""), "gradScaler": USE_SCALER},
         "config": {k: v for k, v in C.items() if k not in ("runId", "label")}, "seed": seed, "params": n_params,
         "metrics": [], "trainCurve": [], "date": time.strftime("%Y-%m-%d")}
    t_build = time.time()
    wins = build_stream(C, steps * B * ctx, rng, R)
    anneal = None
    if C.get("anneal"):
        a_steps = int(round(steps * C["anneal"]["frac"]))
        sub = dict(C, data={"kind": "single", "corpus": C["anneal"]["corpus"]})
        anneal = build_stream(sub, a_steps * B * ctx, np.random.default_rng(seed + 7), {})
        R["anneal"] = {"corpus": C["anneal"]["corpus"], "steps": a_steps}
    R["streamBuildSec"] = round(time.time() - t_build, 1)
    assert len(wins) >= steps * B or anneal is not None, (len(wins), steps * B)

    def get_batch(step, k):
        if anneal is not None and step >= steps - R["anneal"]["steps"]:
            j = (step - (steps - R["anneal"]["steps"])) * B + k * micro
            src = anneal
        else:
            j = step * B + k * micro
            src = wins
        xy = np.stack([np.asarray(w, dtype=np.int64) for w in src[j:j + micro]])
        x = torch.from_numpy(xy[:, :-1]).to(DEV, non_blocking=True)
        y = torch.from_numpy(xy[:, 1:]).to(DEV, non_blocking=True)
        return x, y

    @torch.no_grad()
    def evaluate(name, max_tokens):
        arr, _ = corpus(name)
        n = min(arr.size, max_tokens + 1)
        ws = windows_from(np.asarray(arr[:n]), ctx)
        model.eval(); nll, ntok, nbytes = 0.0, 0, 0
        for i in range(0, len(ws), 32):
            xy = np.stack([w.astype(np.int64) for w in ws[i:i + 32]])
            x = torch.from_numpy(xy[:, :-1]).to(DEV); y_np = xy[:, 1:]
            with autocast():
                logits = model(x)
            nll += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), torch.from_numpy(y_np).to(DEV).reshape(-1), reduction="sum").item()
            ntok += y_np.size; nbytes += int(TOK_BYTES[y_np].sum())
        model.train()
        if ntok == 0:
            return None
        return {"loss": round(nll / ntok, 4), "bpb": round(nll / (math.log(2) * nbytes), 4), "tokens": ntok}

    def lr_at(s):
        peak, wu = C["lr"], C["warmup"]
        if s < wu:
            return peak * (s + 1) / wu
        if C.get("schedule") == "wsd":
            d0 = int(steps * (1 - C["decayFrac"]))
            return peak if s < d0 else peak * max(0.0, (steps - s) / max(1, steps - d0))
        p = (s - wu) / max(1, steps - wu)
        return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": C["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=C["lr"], betas=(0.9, 0.95), fused=DEV == "cuda")
    scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    evals = C.get("evals", [])
    t0 = time.time(); run_loss = 0.0; run_n = 0
    for step in range(steps + 1):
        if step % C["eval_every"] == 0 or step == steps:
            cap = C.get("evalTokensFinal", 600000) if step == steps else C.get("evalTokens", 131072)
            m = {"step": step, "tokens": step * B * ctx, "lr": lr_at(min(step, steps - 1)), "elapsedSec": round(time.time() - t0, 1)}
            for e in evals:
                r = evaluate(e, cap)
                if r:
                    m[e] = r
            if DEV == "cuda":
                m["memGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            R["metrics"].append(m); print(C["runId"], json.dumps(m), flush=True)
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for k in range(accum):
            x, y = get_batch(step, k)
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1)) / accum
            (scaler.scale(loss) if scaler else loss).backward(); tot += loss.item()
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C["clip"])
        if scaler:
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        run_loss += tot; run_n += 1
        if run_n == 25:
            R["trainCurve"].append({"step": step + 1, "trainLoss": round(run_loss / 25, 4)}); run_loss = 0.0; run_n = 0
    R["trainSec"] = round(time.time() - t0, 1)
    if DEV == "cuda":
        R["peakMemGB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if C.get("canaryEval"):
        R["canaryResults"] = canary_eval(model, R)
    if C.get("hellaswag"):
        R["hellaswag"] = hellaswag(model, ctx, C.get("hellaswagMax"))
    if C.get("trainSetEval"):
        # Loss on the repeated training tokens themselves (first 600k), to show the train/val gap
        w = wins[: 600000 // ctx]
        model.eval(); nll, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(w), 32):
                xy = np.stack([np.asarray(v, dtype=np.int64) for v in w[i:i + 32]])
                x = torch.from_numpy(xy[:, :-1]).to(DEV); y = torch.from_numpy(xy[:, 1:]).to(DEV)
                with autocast():
                    lg = model(x)
                nll += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum").item(); n += y.numel()
        R["trainSetLoss"] = round(nll / n, 4)
    with open(os.path.join(out_dir, C["runId"] + ".json"), "w") as f:
        json.dump(R, f, indent=1)
    print("WROTE", C["runId"], json.dumps(R["metrics"][-1])[:400], flush=True)
    del model, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def canary_eval(model, R):
    """Greedy-decode 50 tokens after each canary's 50-token prefix (preceded by EOT, as in the
    stream). Extraction = all 50 tokens match the canary exactly."""
    can = torch.from_numpy(np.load(os.path.join(data_dir, "canaries.npy")).astype(np.int64)).to(DEV)
    plan = R["canaryPlan"]
    idx = [p["index"] for p in plan]
    c = can[idx]
    x = torch.cat([torch.full((c.size(0), 1), EOT, device=DEV, dtype=torch.long), c[:, :50]], 1)
    model.eval()
    for _ in range(50):
        with autocast():
            lg = model(x)[:, -1, :].float()
        x = torch.cat([x, lg.argmax(-1, keepdim=True)], 1)
    gen = x[:, 51:]
    tgt = c[:, 50:100]
    match = (gen == tgt)
    exact = match.all(1).cpu().numpy()
    frac = match.float().mean(1).cpu().numpy()
    lead = []
    for row in match.cpu().numpy():
        k = 0
        while k < 50 and row[k]:
            k += 1
        lead.append(k)
    # Teacher-forced loss on the canary suffix
    xin = torch.cat([torch.full((c.size(0), 1), EOT, device=DEV, dtype=torch.long), c[:, :99]], 1)
    with autocast():
        lg = model(xin).float()
    nll = F.cross_entropy(lg[:, 50:].reshape(-1, lg.size(-1)), c[:, 50:].reshape(-1), reduction="none").view(c.size(0), -1).mean(1).cpu().numpy()
    model.train()
    out = {}
    for p, e, f_, l_, n_ in zip(plan, exact, frac, lead, nll):
        d = out.setdefault(str(p["planned"]), {"n": 0, "exact": 0, "tokenMatch": [], "lead": [], "suffixLoss": [], "seen": []})
        d["n"] += 1; d["exact"] += int(e); d["tokenMatch"].append(round(float(f_), 3)); d["lead"].append(int(l_))
        d["suffixLoss"].append(round(float(n_), 3)); d["seen"].append(p["seen"])
    for d in out.values():
        d["extractionRate"] = round(d["exact"] / d["n"], 4)
        d["meanTokenMatch"] = round(float(np.mean(d["tokenMatch"])), 4)
        d["meanSuffixLoss"] = round(float(np.mean(d["suffixLoss"])), 4)
    print("CANARY", R["runId"], {k: (v["extractionRate"], v["meanTokenMatch"]) for k, v in out.items()}, flush=True)
    return out


@torch.no_grad()
def hellaswag(model, ctx, limit=None):
    items = json.load(open(os.path.join(data_dir, "hellaswag_val_tok.json")))[:limit]
    model.eval(); correct = correct_sum = 0
    for it in items:
        seqs, spans = [], []
        for e in it["endings"]:
            s = (it["ctx"] + e)[-ctx:]
            seqs.append(s); spans.append(len(e) if len(e) < len(s) else len(s) - 1)
        L = max(len(s) for s in seqs)
        x = torch.zeros((4, L), dtype=torch.long, device=DEV)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=DEV)
        with autocast():
            lp = F.log_softmax(model(x).float(), -1)
        scores, sums = [], []
        for i, s in enumerate(seqs):
            n = spans[i]; T = len(s)
            tgt = torch.tensor(s[T - n:], device=DEV)
            v = lp[i, T - n - 1:T - 1].gather(1, tgt[:, None]).sum().item()
            sums.append(v); scores.append(v / n)
        correct += int(np.argmax(scores) == it["label"]); correct_sum += int(np.argmax(sums) == it["label"])
    model.train()
    n = len(items)
    acc = correct / n
    return {"items": n, "accNormByToken": round(acc, 4), "accSum": round(correct_sum / n, 4),
            "ci95": round(1.96 * math.sqrt(acc * (1 - acc) / n), 4)}


for C in QUEUE:
    try:
        run_one(C)
    except Exception as e:  # one failed config must not cost the rest of the queue
        import traceback; traceback.print_exc()
        with open(os.path.join(out_dir, C["runId"] + ".error.txt"), "w") as f:
            f.write(repr(e))
        if DEV == "cuda":
            torch.cuda.empty_cache()
print("QUEUE DONE", flush=True)
