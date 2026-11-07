"""lltDomainAdaptation lab code: data, the evaluation panel, continued pretraining with replay, interpolation,
tokenizer extension init, RAG prompts and a QLoRA task fine-tune. The notebook cells and the page's code blocks
are excerpts of this file. No backslash in this file: NL = chr(10) stands for a newline.
"""
import json, math, os, random, re, string, time, itertools, collections
import torch
import torch.nn.functional as F

NL = chr(10)
MODEL = "Qwen/Qwen3-0.6B-Base"


# ---------------------------------------------------------------- data
def stream_docs(name, split, n, config=None):
    from datasets import load_dataset
    ds = load_dataset(name, name=config, split=split, streaming=True)
    return [{"id": r["id"], "text": r["text"]} for r in itertools.islice(ds, n)]


def cut(text, n_chars):
    # The first n_chars characters, cut back to the last space so no word is split.
    if len(text) <= n_chars:
        return None
    return text[:n_chars].rsplit(" ", 1)[0]


def eval_texts(docs, n_chars, n_docs, tok, max_tokens=1023):
    # Held-out texts: the first n_chars of each long enough doc, kept only if it fits one 1,024-token window.
    out = []
    for d in docs:
        t = cut(d["text"], n_chars)
        if t and len(tok(t, add_special_tokens=False)["input_ids"]) <= max_tokens:
            out.append(t)
        if len(out) == n_docs:
            break
    return out


def load_xquad_hi():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("google/xquad", "xquad.hi/validation-00000-of-00001.parquet", repo_type="dataset"))
    contexts = list(dict.fromkeys(df["context"]))            # 240 paragraphs, in order of first use
    cid = {c: i for i, c in enumerate(contexts)}
    qa = [{"id": r.id, "q": r.question, "a": list(r.answers["text"]), "ctx": cid[r.context]} for r in df.itertuples()]
    return contexts, qa


def load_xstorycloze(lang):
    import csv
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("juletxara/xstory_cloze", "spring2016.val." + lang + ".tsv.split_20_80_eval.tsv", repo_type="dataset")
    rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter=chr(9)))
    items = []
    for r in rows:
        ctx = " ".join(r["input_sentence_" + str(i)] for i in range(1, 5))
        items.append({"ctx": ctx, "endings": [" " + r["sentence_quiz1"], " " + r["sentence_quiz2"]],
                      "label": int(r["answer_right_ending"]) - 1})
    return items


def load_hellaswag(n):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("Rowan/hellaswag", "data/validation-00000-of-00001.parquet", repo_type="dataset"))

    def prep(t):                                            # the lm-eval-harness preprocessing
        t = t.strip().replace(" [title]", ". ")
        t = re.sub(re.escape("[") + ".*?" + re.escape("]"), "", t)
        return t.replace("  ", " ")
    items = []
    for r in df.head(n).itertuples():
        ctx = prep(r.activity_label + ": " + r.ctx_a + " " + r.ctx_b.capitalize())
        items.append({"ctx": ctx, "endings": [" " + prep(e) for e in r.endings], "label": int(r.label)})
    return items


def pack_rows(texts, tok, seq, n_rows):
    # Concatenate documents, each followed by the end-of-text token, and cut into rows of seq tokens.
    rows, buf = [], []
    for t in texts:
        buf += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        while len(buf) >= seq:
            rows.append(buf[:seq]); buf = buf[seq:]
            if len(rows) == n_rows:
                return torch.tensor(rows)
    raise ValueError("not enough text for " + str(n_rows) + " rows, got " + str(len(rows)))


def mix_rows(hi_rows, en_rows, n_rows, replay, seed):
    # A replay fraction r: r of the rows come from general English text, the rest from Hindi, in a seeded random order.
    n_en = round(n_rows * replay)
    src = [1] * n_en + [0] * (n_rows - n_en)
    random.Random(seed).shuffle(src)
    it = [iter(hi_rows), iter(en_rows)]
    return torch.stack([next(it[s]) for s in src]), n_en


# ---------------------------------------------------------------- evaluation panel
@torch.no_grad()
def bits_per_byte(model, tok, texts, bs=2):
    # Score every token of each text after one end-of-text token. bits per byte = total NLL / ln 2 / total UTF-8 bytes.
    dev = next(model.parameters()).device
    nll, ntok = 0.0, 0
    for i in range(0, len(texts), bs):
        seqs = [[tok.eos_token_id] + tok(t, add_special_tokens=False)["input_ids"] for t in texts[i:i + bs]]
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), tok.eos_token_id)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = 1
        x, m = x.to(dev), m.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(input_ids=x, attention_mask=m).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None])[..., 0]
        nll -= float((lp * m[:, 1:]).sum()); ntok += int(m[:, 1:].sum())
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    return {"bpb": nll / math.log(2) / nbytes, "nllPerToken": nll / ntok, "tokens": ntok, "bytes": nbytes}


@torch.no_grad()
def choice_accuracy(model, tok, items, norm_chars=False, bs=16):
    # Pick the ending with the highest log-likelihood (or per character, the acc_norm of lm-eval-harness).
    dev = next(model.parameters()).device
    pairs = [(k, e) for k, it in enumerate(items) for e in it["endings"]]
    scores = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        seqs, spans = [], []
        for k, e in chunk:
            c = tok(items[k]["ctx"], add_special_tokens=False)["input_ids"]
            ce = tok(items[k]["ctx"] + e, add_special_tokens=False)["input_ids"]
            n_common = 0
            while n_common < min(len(c), len(ce)) and c[n_common] == ce[n_common]:
                n_common += 1
            seqs.append(ce); spans.append(max(n_common, 1))
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), tok.eos_token_id)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = 1
        x, m = x.to(dev), m.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(input_ids=x, attention_mask=m).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, x[:, 1:, None])[..., 0]
        for j, s in enumerate(seqs):
            sc = float(lp[j, spans[j] - 1:len(s) - 1].sum())
            if norm_chars:
                sc /= len(chunk[j][1])
            scores.append(sc)
    correct, p = 0, 0
    for it in items:
        n = len(it["endings"])
        s = scores[p:p + n]; p += n
        correct += int(max(range(n), key=lambda q: s[q]) == it["label"])
    return correct / len(items), correct


def panel(model, tok, E, with_tasks=True):
    # The fixed evaluation panel: Hindi and English held-out bits per byte, plus three small benchmarks.
    r = {"hiBpb": bits_per_byte(model, tok, E["hi"])["bpb"], "enBpb": bits_per_byte(model, tok, E["en"])["bpb"]}
    if with_tasks:
        r["xscHi"], r["xscHiCorrect"] = choice_accuracy(model, tok, E["xsc_hi"])
        r["xscEn"], r["xscEnCorrect"] = choice_accuracy(model, tok, E["xsc_en"])
        r["hellaswag"], r["hellaswagCorrect"] = choice_accuracy(model, tok, E["hellaswag"], norm_chars=True)
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}


# ---------------------------------------------------------------- question answering, closed book and RAG
def qa_prompt(q, contexts=(), shots=()):
    parts = ["प्रश्न: " + sq + NL + "उत्तर: " + sa for sq, sa in shots]
    parts += ["संदर्भ: " + c for c in contexts]
    parts.append("प्रश्न: " + q + NL + "उत्तर:")
    return (NL + NL).join(parts)


def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation + "।"))
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def answer(model, tok, prompts, max_new=40, bs=8):
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip().split(NL)[0].strip())
    tok.padding_side = "right"
    return outs


def qa_scores(preds, qa):
    em = sum(max(float(normalize(p) == normalize(a)) for a in x["a"]) for p, x in zip(preds, qa)) / len(qa)
    f = sum(max(f1(p, a) for a in x["a"]) for p, x in zip(preds, qa)) / len(qa)
    return {"em": round(em, 4), "f1": round(f, 4), "n": len(qa)}


def qa_eval(model, tok, Q, modes=("closed0", "closed3", "rag0", "rag3")):
    # Q holds the test questions, the 3 example question-answer pairs, the paragraph pool and the retrieved ids.
    out = {}
    for mode in modes:
        shots = Q["shots"] if mode.endswith("3") else ()
        prompts = []
        for x in Q["test"]:
            ctx = [Q["pool"][i] for i in Q["retrieved"][x["id"]][:2]] if mode.startswith("rag") else ()
            prompts.append(qa_prompt(x["q"], ctx, shots))
        preds = answer(model, tok, prompts)
        out[mode] = qa_scores(preds, Q["test"])
        out[mode]["examples"] = [{"q": x["q"], "gold": x["a"][0], "pred": p} for x, p in list(zip(Q["test"], preds))[:5]]
    return out


# ---------------------------------------------------------------- continued pretraining
def lr_at(step, steps, peak, warmup, min_ratio=0.1):
    # Re-warm linearly to the peak, then re-decay with a cosine to min_ratio of the peak.
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def train_cpt(model, rows, cfg, on_eval=None):
    import bitsandbytes as bnb
    dev = next(model.parameters()).device
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = bnb.optim.AdamW8bit([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=cfg["peak_lr"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg.get("init_scale", 65536.0))
    steps, per_step, micro = cfg["steps"], cfg["rows_per_step"], cfg["micro"]
    warmup = max(1, round(steps * cfg.get("warmup_frac", 0.04)))
    log, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for s in range(steps):
        lr = lr_at(s, steps, cfg["peak_lr"], warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = rows[s * per_step:(s + 1) * per_step]
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, per_step, micro):
            x = batch[i:i + micro].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, labels=x).loss
            frac = x.shape[0] / per_step
            tot += float(loss.detach()) * frac
            scaler.scale(loss * frac).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt); scaler.update()
        rec = {"step": s + 1, "tokens": (s + 1) * per_step * rows.shape[1], "trainLoss": round(tot, 4), "lr": lr,
               "gradNorm": round(gn, 3), "scale": scaler.get_scale(), "sec": round(time.time() - t0, 1)}
        if on_eval and ((s + 1) % cfg["eval_every"] == 0 or s + 1 == steps):
            opt.zero_grad(set_to_none=True)          # free the gradients before the eval forward passes
            model.eval(); rec.update(on_eval(model)); model.train()
        log.append(rec)
        if (s + 1) % 25 == 0 or s < 3:
            print(json.dumps(rec), flush=True)
    model.eval()
    model.config.use_cache = True                  # generation needs the KV cache back on
    return log, round(torch.cuda.max_memory_allocated() / 1e9, 2), round(time.time() - t0, 1)


def interpolate(model, base_sd, ft_sd, a):
    # WiSE-FT: theta(a) = (1 - a) * theta_base + a * theta_ft, parameter by parameter.
    sd = {k: base_sd[k] + a * (ft_sd[k] - base_sd[k]) for k in ft_sd}
    model.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------- tokenizer extension
def add_tokens_mean_init(model, n_vocab, parts):
    # Grow the (tied) embedding, then set each new row to the mean of the rows of the tokens it was made of.
    model.resize_token_embeddings(n_vocab, pad_to_multiple_of=64, mean_resizing=False)
    E = model.get_input_embeddings().weight
    with torch.no_grad():
        for new_id, old_ids in parts.items():
            E[int(new_id)] = E[old_ids].mean(0)
    return model


# ---------------------------------------------------------------- LoRA task fine-tune
def qa_train_examples(tok, Q, seed=0):
    # Prompt = gold paragraph + one other paragraph, in random order, then the question. Loss only on the answer.
    rng = random.Random(seed)
    ex = []
    train_ctx = sorted({x["ctx"] for x in Q["train"]})
    for x in Q["train"]:
        other = rng.choice([c for c in train_ctx if c != x["ctx"]])
        ctx = [Q["contexts"][x["ctx"]], Q["contexts"][other]]
        rng.shuffle(ctx)
        p = tok(qa_prompt(x["q"], ctx), add_special_tokens=False)["input_ids"]
        a = tok(" " + x["a"][0] + NL, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ex.append((p + a, [-100] * len(p) + a))
    return ex


def train_lora_qa(qm, tok, ex, epochs=2, bs=8, micro=2, lr=2e-4, seed=0, max_steps=None):
    # Masked-loss fine-tune of the trainable (LoRA) parameters: batches of bs examples, micro-batches of micro.
    tr = [p for p in qm.parameters() if p.requires_grad]
    order = list(range(len(ex))) * epochs
    random.Random(seed).shuffle(order)
    steps = len(order) // bs if max_steps is None else max_steps
    opt = torch.optim.AdamW(tr, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    qm.train(); qm.config.use_cache = False
    log, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s, steps, lr, max(1, steps // 20), min_ratio=0.0)
        idx = order[s * bs:(s + 1) * bs]
        n_ans = sum(sum(1 for v in ex[i][1] if v != -100) for i in idx)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for c in range(0, bs, micro):
            part = idx[c:c + micro]
            Lm = max(len(ex[i][0]) for i in part)
            x = torch.full((len(part), Lm), tok.eos_token_id)
            y = torch.full((len(part), Lm), -100)
            m = torch.zeros((len(part), Lm), dtype=torch.long)
            for k, i in enumerate(part):
                ids, lab = ex[i]
                x[k, :len(ids)] = torch.tensor(ids); y[k, :len(lab)] = torch.tensor(lab); m[k, :len(ids)] = 1
            w = sum(sum(1 for v in ex[i][1] if v != -100) for i in part) / n_ans   # weight by answer tokens
            with torch.autocast("cuda", dtype=torch.float16):
                loss = qm(input_ids=x.cuda(), attention_mask=m.cuda(), labels=y.cuda()).loss
            tot += float(loss.detach()) * w
            scaler.scale(loss * w).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(tr, 1.0)
        scaler.step(opt); scaler.update()
        log.append({"step": s + 1, "trainLoss": round(tot, 4), "sec": round(time.time() - t0, 1)})
        if (s + 1) % 30 == 0 or s < 2:
            print(json.dumps(log[-1]), flush=True)
    qm.eval(); qm.config.use_cache = True
    return log, round(torch.cuda.max_memory_allocated() / 1e9, 2), round(time.time() - t0, 1)
