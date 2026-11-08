"""lltRewardModels Run Library worker. One job per process.

Usage: python worker_rm.py <job.json> <result.json>
job.kind:
  sample     best-of-n pool: n completions per prompt from SmolLM2-360M-Instruct (fp32 sampling)
  gold       score the pool, held-out UltraFeedback pairs and a RewardBench 2 subset with the gold RM
  train      Bradley-Terry reward model, written by hand, SmolLM2-360M + scalar head, full fine-tune
  trlcheck   own loss vs TRL RewardTrainer.compute_loss on the same batch, then a short RewardTrainer run
All data selection is deterministic (see select_pairs, select_prompts, rb2_subset).
"""
import gc, json, math, os, random, sys, time, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
job = json.load(open(sys.argv[1])); OUT = sys.argv[2]
R = {"tag": job["tag"], "kind": job["kind"], "job": job}
WORK = job.get("work", "/kaggle/working")


def save():
    json.dump(R, open(OUT, "w"), indent=1, default=str)


import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BASE = "HuggingFaceTB/SmolLM2-360M"
POLICY = "HuggingFaceTB/SmolLM2-360M-Instruct"
GOLD = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
UF = "HuggingFaceH4/ultrafeedback_binarized"
MAXLEN = 1024


def peak():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


CUDA = torch.cuda.is_available()


def reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def sync():
    if CUDA:
        torch.cuda.synchronize()


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rm_tokenizer():
    """SmolLM2 base ships no chat template: borrow the instruct one, keep pad apart from the turn end."""
    tok = AutoTokenizer.from_pretrained(BASE)
    inst = AutoTokenizer.from_pretrained(POLICY)
    tok.chat_template = inst.chat_template
    tok.pad_token = "<|endoftext|>"
    return tok


def conv(prompt, response):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def ids_of(tok, prompt, response):
    return list(tok.apply_chat_template(conv(prompt, response))["input_ids"])


def select_pairs(tok, split, n, seed=0, balanced=False, start=0):
    """Shuffle the split with a fixed seed, keep pairs whose two sequences fit MAXLEN tokens, take n.
    balanced=True takes n/2 pairs whose chosen is longer and n/2 whose chosen is shorter (equal lengths skipped)."""
    from datasets import load_dataset
    ds = load_dataset(UF, split=split)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    out, longer, shorter, read = [], 0, 0, 0
    for i in range(start, len(ds)):
        ex = ds[i]; read += 1
        p = ex["prompt"]; c = ex["chosen"][-1]["content"]; r = ex["rejected"][-1]["content"]
        if c == r:
            continue
        ci = ids_of(tok, p, c); ri = ids_of(tok, p, r)
        if len(ci) > MAXLEN or len(ri) > MAXLEN:
            continue
        row = dict(prompt=p, c=ci, r=ri, sc=ex["score_chosen"], sr=ex["score_rejected"], pid=ex["prompt_id"])
        if balanced:
            if len(ci) == len(ri):
                continue
            if len(ci) > len(ri):
                if longer >= n // 2:
                    continue
                longer += 1
            else:
                if shorter >= n // 2:
                    continue
                shorter += 1
        out.append(row)
        if len(out) >= n:
            break
    return out, read


def select_prompts(n, max_prompt_tokens=200, max_ref_tokens=160):
    """First n test_prefs prompts (file order) whose prompt fits max_prompt_tokens and whose chosen reference answer
    fits max_ref_tokens, so most samples can finish inside the generation budget."""
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(POLICY)
    ds = load_dataset(UF, split="test_prefs")
    out = []
    for i, ex in enumerate(ds):
        m = [{"role": "user", "content": ex["prompt"]}]
        L = len(tok.apply_chat_template(m, add_generation_prompt=True)["input_ids"])
        ref = len(tok(ex["chosen"][-1]["content"], add_special_tokens=False).input_ids)
        if L <= max_prompt_tokens and ref <= max_ref_tokens:
            out.append(dict(row=i, prompt=ex["prompt"], prompt_id=ex["prompt_id"], promptTokens=L, refTokens=ref))
        if len(out) >= n:
            break
    return out


def rb2_subset(per_subset, seed=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/reward-bench-2", split="test")
    rng = random.Random(seed)
    out = []
    for sub in ["Factuality", "Precise IF", "Math", "Safety", "Focus"]:
        idx = [i for i, s in enumerate(ds["subset"]) if s == sub]
        pick = sorted(rng.sample(idx, min(per_subset, len(idx))))
        for i in pick:
            ex = ds[i]
            out.append(dict(id=ex["id"], subset=sub, prompt=ex["prompt"], chosen=list(ex["chosen"]), rejected=list(ex["rejected"])))
    return out


def score_seqs(model, seqs, pad_id, max_tokens=16384, max_batch=32, dtype=torch.float16):
    """Score token-id lists; returns rewards in the input order. Batches sorted by length, right padding."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    out = [None] * len(seqs); i = 0
    while i < len(order):
        L = len(seqs[order[i]])
        b = max(1, min(max_batch, max_tokens // max(L, 1)))
        idx = order[i:i + b]
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long); att = torch.zeros_like(ids)
        for k, j in enumerate(idx):
            s = seqs[j]; ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            r = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.squeeze(-1).float().cpu().tolist()
        for k, j in enumerate(idx):
            out[j] = r[k]
        i += b
    return out


def pair_metrics(rc, rr, pairs):
    rc = np.array(rc); rr = np.array(rr)
    win = (rc > rr).astype(float)
    lc = np.array([len(p["c"]) for p in pairs]); lr = np.array([len(p["r"]) for p in pairs])
    tie = np.array([p["sc"] == p["sr"] for p in pairs])
    m = {"n": len(pairs), "acc": float(win.mean()), "accNoScoreTie": float(win[~tie].mean()), "nScoreTie": int(tie.sum()),
         "accChosenLonger": float(win[lc > lr].mean()), "nChosenLonger": int((lc > lr).sum()),
         "accChosenShorter": float(win[lc < lr].mean()), "nChosenShorter": int((lc < lr).sum()),
         "lengthOnlyAcc": float((lc > lr).mean()), "meanMargin": float((rc - rr).mean())}
    allr = np.concatenate([rc, rr]); alll = np.concatenate([lc, lr])
    m["pearsonRewardLength"] = float(np.corrcoef(allr, alll)[0, 1])
    return m, win.tolist()


def load_pool():
    p = job.get("pool") or os.path.join(WORK, "pool.json")
    return json.load(open(p))


try:
    kind = job["kind"]
    R["env"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    if kind == "sample":
        tok = AutoTokenizer.from_pretrained(POLICY); tok.padding_side = "left"
        prompts = select_prompts(job["prompts"], max_ref_tokens=job.get("maxRefTokens", 160))
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(POLICY, dtype=torch.float32, device_map=DEV).eval()
        R["loadSec"] = round(time.time() - t0, 1)
        n = job["n"]; per = job["promptsPerBatch"]; new = job["newTokens"]
        torch.manual_seed(job.get("seed", 0))
        pool = {"policy": POLICY, "dtype": "float32", "n": n, "newTokens": new, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "prompts": prompts, "samples": []}
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        reset_peak(); times = []
        for b in range(0, len(prompts), per):
            group = prompts[b:b + per]
            msgs = [tok.apply_chat_template([{"role": "user", "content": g["prompt"]}], tokenize=False, add_generation_prompt=True)
                    for g in group for _ in range(n)]
            enc = tok(msgs, return_tensors="pt", padding=True).to(DEV)
            sync(); t = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=new, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     pad_token_id=tok.pad_token_id, eos_token_id=im_end)
            sync(); times.append(time.time() - t)
            gen = out[:, enc.input_ids.shape[1]:].cpu()
            for k in range(gen.shape[0]):
                g = gen[k].tolist()
                ended = im_end in g
                if ended:
                    g = g[:g.index(im_end)]
                while g and g[-1] == tok.pad_token_id:
                    g = g[:-1]
                pool["samples"].append(dict(p=b + k // n, j=k % n, text=tok.decode(g, skip_special_tokens=True),
                                            newTokens=len(g), ended=ended))
            print("batch", b // per, "sec", round(times[-1], 1), "peak", peak(), flush=True)
        json.dump(pool, open(os.path.join(WORK, "pool.json"), "w"))
        ends = [s["ended"] for s in pool["samples"]]
        R.update(ok=True, prompts=len(prompts), completions=len(pool["samples"]), batchSec=[round(x, 1) for x in times],
                 totalGenSec=round(sum(times), 1), peakMemGB=peak(), endedShare=float(np.mean(ends)),
                 meanNewTokens=float(np.mean([s["newTokens"] for s in pool["samples"]])))

    elif kind == "gold":
        t0 = time.time()
        GOLD = job.get("goldModel", GOLD)
        gtok = AutoTokenizer.from_pretrained(GOLD)
        gold = AutoModelForSequenceClassification.from_pretrained(GOLD, num_labels=1, dtype=torch.float16, device_map=DEV).eval()
        if gtok.pad_token is None:
            gtok.pad_token = gtok.eos_token
        gold.config.pad_token_id = gtok.pad_token_id
        gold.config.get_text_config().pad_token_id = gtok.pad_token_id
        R["loadSec"] = round(time.time() - t0, 1); R["residentGB"] = round(torch.cuda.memory_allocated() / 1e9, 2) if CUDA else None
        R["padToken"] = gtok.pad_token; R["bos"] = gtok.bos_token

        def gids(prompt, response, cap=job.get("goldMaxTokens", 4096)):
            s = gtok.apply_chat_template(conv(prompt, response), tokenize=False)
            if gtok.bos_token is not None and s.startswith(gtok.bos_token):
                s = s[len(gtok.bos_token):]
            ids = gtok(s, add_special_tokens=False).input_ids
            return ids[:cap], len(ids) > cap

        R["exampleRendered"] = gtok.apply_chat_template(conv("Name a fruit.", "A mango."), tokenize=False)
        reset_peak()
        # 1. the best-of-n pool
        pool = load_pool()
        seqs = [gids(pool["prompts"][s["p"]]["prompt"], s["text"])[0] for s in pool["samples"]]
        t = time.time(); g = score_seqs(gold, seqs, gtok.pad_token_id, max_tokens=job.get("goldTokBudget", 12288)); R["poolSec"] = round(time.time() - t, 1)
        json.dump({"gold": GOLD, "scores": g}, open(os.path.join(WORK, "gold_pool.json"), "w"))
        R["poolSeqPerSec"] = round(len(seqs) / R["poolSec"], 2)
        # 2. held-out UltraFeedback pairs (same selection as the proxy's held-out set), chat format
        rtok = rm_tokenizer()
        held, read = select_pairs(rtok, "test_prefs", job["heldPairs"], seed=None)
        cs, rs = [], []; capped = 0
        from datasets import load_dataset
        te = load_dataset(UF, split="test_prefs"); byid = {}
        for ex in te:
            byid.setdefault(ex["prompt_id"], ex)
        for h in held:
            ex = byid[h["pid"]]
            a, ca = gids(ex["prompt"], ex["chosen"][-1]["content"]); b, cb = gids(ex["prompt"], ex["rejected"][-1]["content"])
            cs.append(a); rs.append(b); capped += int(ca) + int(cb)
        t = time.time(); sc = score_seqs(gold, cs + rs, gtok.pad_token_id, max_tokens=job.get("goldTokBudget", 12288)); R["heldSec"] = round(time.time() - t, 1)
        m, win = pair_metrics(sc[:len(held)], sc[len(held):], held)
        R["heldOut"] = m; R["heldOutCapped"] = capped
        json.dump({"gold": GOLD, "pid": [h["pid"] for h in held], "rc": sc[:len(held)], "rr": sc[len(held):], "win": win},
                  open(os.path.join(WORK, "gold_held.json"), "w"))
        # 3. RewardBench 2 subset, chosen must beat all three rejected
        items = rb2_subset(job["rb2PerSubset"])
        flat, owner = [], []
        for k, it in enumerate(items):
            for resp in it["chosen"][:1] + it["rejected"][:3]:
                flat.append(gids(it["prompt"], resp, cap=job.get("rb2MaxTokens", 2048))[0]); owner.append(k)
        t = time.time(); sc = score_seqs(gold, flat, gtok.pad_token_id, max_tokens=job.get("goldTokBudget", 12288)); R["rb2Sec"] = round(time.time() - t, 1)
        res = {}
        for k, it in enumerate(items):
            s = sc[4 * k:4 * k + 4]
            res.setdefault(it["subset"], []).append(float(s[0] > max(s[1:])))
        R["rb2"] = {sub: {"n": len(v), "acc": float(np.mean(v))} for sub, v in res.items()}
        R["rb2Mean"] = float(np.mean([v["acc"] for v in R["rb2"].values()]))
        json.dump({"gold": GOLD, "ids": [it["id"] for it in items], "subset": [it["subset"] for it in items], "scores": sc},
                  open(os.path.join(WORK, "gold_rb2.json"), "w"))
        R.update(ok=True, peakMemGB=peak(), totalSec=round(time.time() - t0, 1))

    elif kind == "train":
        seed = job["seed"]; tag = job["tag"]
        tok = rm_tokenizer()
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        t0 = time.time()
        train, read = select_pairs(tok, "train_prefs", job["pairs"], seed=0, balanced=job.get("balanced", False))
        held, _ = select_pairs(tok, "test_prefs", job["heldPairs"], seed=None)
        R["data"] = {"trainPairs": len(train), "rowsRead": read, "heldPairs": len(held), "selectSec": round(time.time() - t0, 1),
                     "trainChosenLongerShare": float(np.mean([len(p["c"]) > len(p["r"]) for p in train])),
                     "trainScoreTieShare": float(np.mean([p["sc"] == p["sr"] for p in train])),
                     "trainMeanTokens": float(np.mean([len(p["c"]) + len(p["r"]) for p in train]) / 2)}
        print(R["data"], flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.float32, device_map=DEV)
        model.config.pad_token_id = tok.pad_token_id
        model.config.get_text_config().pad_token_id = tok.pad_token_id
        R["head"] = {"name": "score", "shape": list(model.score.weight.shape), "bias": model.score.bias is not None,
                     "initStd": float(model.score.weight.std())}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        params = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=0.0)
        B = job["batchPairs"]; micro = job.get("microPairs", B)
        # bucketing: shuffle, sort inside chunks of 64 pairs by length, cut into batches, shuffle the batch order
        rng = random.Random(seed)
        order = list(range(len(train))); rng.shuffle(order)
        batches = []
        for c0 in range(0, len(order), 64):
            chunk = sorted(order[c0:c0 + 64], key=lambda i: -max(len(train[i]["c"]), len(train[i]["r"])))
            batches += [chunk[k:k + B] for k in range(0, len(chunk), B)]
        rng.shuffle(batches)
        steps = len(batches); warm = max(1, int(0.03 * steps))
        def lr_factor(s):   # linear warmup over 3% of the steps, then cosine to zero
            if s < warm:
                return (s + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        scaler = torch.amp.GradScaler("cuda", init_scale=2048, enabled=CUDA)
        evals_at = sorted(set(int(round(f * steps)) for f in job.get("evalFractions", [1.0])))
        pool = load_pool() if job.get("scorePool") else None
        pool_seqs = None
        if pool:
            pool_seqs = [ids_of(tok, pool["prompts"][s["p"]]["prompt"], s["text"]) for s in pool["samples"]]
            R["poolSeqs"] = len(pool_seqs)
        metrics, evals = [], []

        def evaluate(step):
            model.eval(); t = time.time()
            sc = score_seqs(model, [h["c"] for h in held] + [h["r"] for h in held], tok.pad_token_id)
            m, win = pair_metrics(sc[:len(held)], sc[len(held):], held)
            e = {"step": step, "pairsSeen": min(step * B, len(train)), "heldOut": m, "evalSec": round(time.time() - t, 1)}
            json.dump({"pid": [h["pid"] for h in held], "rc": sc[:len(held)], "rr": sc[len(held):], "win": win},
                      open(os.path.join(WORK, "proxy_held_%s_step%d.json" % (tag, step)), "w"))
            if pool_seqs is not None:
                t2 = time.time(); ps = score_seqs(model, pool_seqs, tok.pad_token_id)
                json.dump({"proxy": tag, "step": step, "scores": ps}, open(os.path.join(WORK, "proxy_pool_%s_step%d.json" % (tag, step)), "w"))
                e["poolSec"] = round(time.time() - t2, 1)
            evals.append(e); R["evals"] = evals; save()
            print("EVAL", json.dumps(e)[:400], flush=True)
            model.train()

        if job.get("evalAtZero"):
            evaluate(0)
        model.train(); reset_peak(); tstart = time.time(); nonfinite = 0
        for s, bidx in enumerate(batches):
            t1 = time.time()
            chunk = [train[i] for i in bidx]
            while True:
                try:
                    opt.zero_grad(set_to_none=True); tl = ta = tm = 0.0
                    for k in range(0, len(chunk), micro):
                        sub = chunk[k:k + micro]
                        seqs = [p["c"] for p in sub] + [p["r"] for p in sub]
                        L = max(len(x) for x in seqs)
                        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(ids)
                        for q, x in enumerate(seqs):
                            ids[q, :len(x)] = torch.tensor(x); att[q, :len(x)] = 1
                        with torch.autocast("cuda", dtype=torch.float16, enabled=CUDA):
                            rew = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.squeeze(-1).float()
                        rc, rr = rew[:len(sub)], rew[len(sub):]
                        loss = -torch.nn.functional.logsigmoid(rc - rr).mean() * (len(sub) / len(chunk))
                        scaler.scale(loss).backward()
                        tl += loss.item(); ta += (rc > rr).float().sum().item(); tm += (rc - rr).sum().item()
                    break
                except torch.OutOfMemoryError:
                    R.setdefault("oomRetries", []).append({"step": s + 1, "micro": micro})
                    opt.zero_grad(set_to_none=True); free()
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            if not math.isfinite(gn):
                nonfinite += 1
            scaler.step(opt); scaler.update(); sched.step()
            sync()
            metrics.append({"step": s + 1, "pairs": (s + 1) * B, "trainLoss": round(tl, 5), "trainAcc": round(ta / len(chunk), 4),
                            "margin": round(tm / len(chunk), 4), "gradNorm": round(gn, 4) if math.isfinite(gn) else None,
                            "lr": sched.get_last_lr()[0], "sec": round(time.time() - t1, 3), "maxLen": max(max(len(p["c"]), len(p["r"])) for p in chunk),
                            "scale": scaler.get_scale()})
            if (s + 1) % 25 == 0 or s == 0:
                print(json.dumps(metrics[-1]), "elapsed", round(time.time() - tstart), flush=True)
            if (s + 1) in evals_at:
                R["trainSecSoFar"] = round(time.time() - tstart, 1)
                evaluate(s + 1)
            if job.get("maxMinutes") and time.time() - tstart > 60 * job["maxMinutes"]:
                R["stoppedEarlyAtStep"] = s + 1; break
        R["metrics"] = metrics
        R["trainSec"] = round(sum(m["sec"] for m in metrics), 1)
        R["sPerStepMedian"] = float(np.median([m["sec"] for m in metrics[2:]])) if len(metrics) > 3 else None
        R["peakMemGB"] = peak(); R["nonFiniteGradSteps"] = nonfinite
        if job.get("rb2PerSubset"):
            model.eval()
            items = rb2_subset(job["rb2PerSubset"]); flat = []
            for it in items:
                for resp in it["chosen"][:1] + it["rejected"][:3]:
                    flat.append(ids_of(tok, it["prompt"], resp)[:job.get("rb2MaxTokens", 2048)])
            sc = score_seqs(model, flat, tok.pad_token_id)
            res = {}
            for k, it in enumerate(items):
                v = sc[4 * k:4 * k + 4]
                res.setdefault(it["subset"], []).append(float(v[0] > max(v[1:])))
            R["rb2"] = {sub: {"n": len(v), "acc": float(np.mean(v))} for sub, v in res.items()}
            R["rb2Mean"] = float(np.mean([v["acc"] for v in R["rb2"].values()]))
            json.dump({"proxy": tag, "ids": [it["id"] for it in items], "subset": [it["subset"] for it in items], "scores": sc},
                      open(os.path.join(WORK, "proxy_rb2_%s.json" % tag), "w"))
        if job.get("saveDir"):
            d = os.path.join(WORK, job["saveDir"])
            model.config.use_cache = True
            model.half().save_pretrained(d, safe_serialization=True)
            tok.save_pretrained(d)
            R["saved"] = sorted(os.listdir(d))
        R["ok"] = True

    elif kind == "trlcheck":
        from datasets import Dataset
        from trl import RewardConfig, RewardTrainer
        import trl
        R["trl"] = trl.__version__
        tok = rm_tokenizer()
        rows, _ = select_pairs(tok, "train_prefs", job["pairs"], seed=0)
        from datasets import load_dataset
        tr = load_dataset(UF, split="train_prefs").shuffle(seed=0)
        keep = set(p["pid"] for p in rows); recs = []
        for ex in tr:
            if ex["prompt_id"] in keep and ex["chosen"][-1]["content"] != ex["rejected"][-1]["content"]:
                recs.append({"chosen": conv(ex["prompt"], ex["chosen"][-1]["content"]),
                             "rejected": conv(ex["prompt"], ex["rejected"][-1]["content"])})
            if len(recs) >= len(rows):
                break
        torch.manual_seed(0)
        model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.float32, device_map=DEV)
        cfg = RewardConfig(output_dir=os.path.join(WORK, "trl_rm"), max_length=MAXLEN, per_device_train_batch_size=job["batchPairs"],
                           gradient_accumulation_steps=1, max_steps=job["steps"], learning_rate=job["lr"], fp16=True, bf16=False,
                           lr_scheduler_type="cosine", warmup_steps=0.03,
                           gradient_checkpointing=True, report_to="none", logging_steps=1, save_strategy="no", seed=0, data_seed=0)
        trainer = RewardTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(recs), processing_class=tok)
        R["trainableDtypes"] = sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad})
        R["trlKeptPairs"] = len(trainer.train_dataset); R["pairsGiven"] = len(recs)
        own = [(ids_of(tok, r["chosen"][0]["content"], r["chosen"][1]["content"]), ids_of(tok, r["rejected"][0]["content"], r["rejected"][1]["content"])) for r in recs]
        R["idsEqual"] = all(list(trainer.train_dataset[i]["chosen_ids"]) == own[i][0] and list(trainer.train_dataset[i]["rejected_ids"]) == own[i][1]
                            for i in range(len(trainer.train_dataset)))
        batch = next(iter(trainer.get_train_dataloader()))
        batch = {k: v.to(DEV) for k, v in batch.items()}
        trainer.model.eval()
        with torch.no_grad():
            lib = trainer.compute_loss(trainer.model, dict(batch)).item()
            h = trainer.model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            last = batch["attention_mask"].sum(-1) - 1                       # right padding: the last real token
            r = trainer.model.score(h[torch.arange(h.shape[0]), last]).squeeze(-1).float()
            n = r.shape[0] // 2
            ownl = -torch.nn.functional.logsigmoid(r[:n] - r[n:]).mean().item()
        R["lossCheck"] = {"batchShape": list(batch["input_ids"].shape), "trl": lib, "own": ownl, "absDiff": abs(lib - ownl)}
        print(R["lossCheck"], flush=True)
        trainer.model.train(); reset_peak(); t = time.time()
        out = trainer.train()
        R["trainSec"] = round(time.time() - t, 1); R["peakMemGB"] = peak()
        R["log"] = [h for h in trainer.state.log_history if "loss" in h]
        R["ok"] = True

    save()
except Exception as e:
    R["ok"] = False; R["error"] = repr(e)[:2000]; R["traceback"] = traceback.format_exc()[-4000:]
    save(); print(R["traceback"], flush=True)
