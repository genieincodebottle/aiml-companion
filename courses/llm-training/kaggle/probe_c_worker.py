"""Batch C probe worker. One job per process. Usage: python probe_c_worker.py <job.json> <result.json>
job.kind in: rm | dpo | grpo | toolsft | kd | kd_onpolicy | trl_rm | trl_dpo | trl_grpo | trl_distill
"""
import gc, json, os, sys, time, traceback, statistics, io, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
job = json.load(open(sys.argv[1])); outp = sys.argv[2]
R = {"tag": job["tag"], "kind": job["kind"], "model": job.get("model")}
def save():
    json.dump(R, open(outp, "w"), indent=1, default=str)
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
logbuf = io.StringIO()
_h = logging.StreamHandler(logbuf); _h.setLevel(logging.WARNING)
logging.getLogger("transformers").addHandler(_h)
DEV = "cuda:0"

def peak():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)
def med(xs, skip=2):
    xs = xs[skip:] if len(xs) > skip + 1 else xs
    return round(statistics.median(xs), 3) if xs else None
def free():
    gc.collect(); torch.cuda.empty_cache()

def load_tok(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def lora_model(model, task="CAUSAL_LM", r=16):
    from peft import LoraConfig, get_peft_model
    names = [n for n, _ in model.named_modules()]
    ex = r".*(visual|vision).*" if any(("visual" in n or "vision" in n) for n in names) else None
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", exclude_modules=ex, task_type=task)
    m = get_peft_model(model, cfg)
    for p in m.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return m

def prefs(tok, n, max_len):
    """UltraFeedback binarized pairs -> list of (prompt_ids, chosen_ids, rejected_ids) with total length <= max_len."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=f"train_prefs[:{n * 3}]")
    out = []
    for ex in ds:
        p = tok(ex["prompt"], add_special_tokens=False).input_ids[:200]
        c = tok("\n\n" + ex["chosen"][-1]["content"], add_special_tokens=False).input_ids
        r = tok("\n\n" + ex["rejected"][-1]["content"], add_special_tokens=False).input_ids
        c = c[: max_len - len(p)]; r = r[: max_len - len(p)]
        if len(c) < 30 or len(r) < 30:
            continue
        out.append((p, c, r))
        if len(out) >= n:
            break
    return out

def pad_batch(seqs, pad, L):
    ids = torch.full((len(seqs), L), pad, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:L]; ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    return ids, att

def seq_logp(model, ids, att, plen):
    """sum of token log-probs over the response part (positions >= plen) for each row. Computes log-softmax in chunks of rows."""
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=att).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    pos = torch.arange(lp.shape[1], device=lp.device).unsqueeze(0)
    mask = (pos >= (plen.unsqueeze(1) - 1)) & att[:, 1:].bool()
    return (lp * mask).sum(-1), mask.sum(-1)

def timed(fn):
    torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize(); return r, time.time() - t

try:
    kind = job["kind"]; name = job.get("model")
    T = job.get("seq", 512)
    if kind == "rm":
        from transformers import AutoModelForSequenceClassification
        tok = load_tok(name)
        data = prefs(tok, job["steps"] * job["pairs"], T)
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, dtype=torch.float32, device_map=DEV)
        model.config.pad_token_id = tok.pad_token_id
        try: model.config.get_text_config().pad_token_id = tok.pad_token_id
        except Exception: pass
        R["loadedClass"] = type(model).__name__; R["loadSec"] = round(time.time() - t0, 1)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        model.config.use_cache = False
        if job["mode"] == "lora":
            model = lora_model(model, "SEQ_CLS")
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        else:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        tr = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in tr)
        model.train(); scaler = torch.amp.GradScaler("cuda", init_scale=2048)
        torch.cuda.reset_peak_memory_stats(); times = []; losses = []; accs = []; micro = job["pairs"]
        for s in range(job["steps"]):
            chunk = data[s * job["pairs"]:(s + 1) * job["pairs"]]
            t1 = time.time()
            while True:
                try:
                    opt.zero_grad(set_to_none=True); tl = 0; ta = 0
                    for i in range(0, len(chunk), micro):
                        sub = chunk[i:i + micro]
                        seqs = [p + c for p, c, r in sub] + [p + r for p, c, r in sub]
                        ids, att = pad_batch(seqs, tok.pad_token_id, T); ids = ids.cuda(); att = att.cuda()
                        with torch.autocast("cuda", dtype=torch.float16):
                            rew = model(input_ids=ids, attention_mask=att).logits.squeeze(-1).float()
                        rc, rr = rew[:len(sub)], rew[len(sub):]
                        loss = -torch.nn.functional.logsigmoid(rc - rr).mean() * (len(sub) / len(chunk))
                        scaler.scale(loss).backward(); tl += loss.item(); ta += (rc > rr).float().sum().item()
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
                    break
                except torch.OutOfMemoryError:
                    R.setdefault("oomRetries", []).append({"step": s, "micro": micro}); opt.zero_grad(set_to_none=True); free()
                    if micro == 1: raise
                    micro = max(1, micro // 2)
            torch.cuda.synchronize(); times.append(time.time() - t1); losses.append(tl); accs.append(ta / len(chunk))
            print(f"step {s+1} loss {tl:.4f} acc {accs[-1]:.2f} {times[-1]:.2f}s", flush=True)
        R.update(stepsDone=len(times), pairsPerStep=job["pairs"], microPairs=micro, seq=T, sPerStepMedian=med(times), peakMemGB=peak(),
                 seqPerSec=round(2 * job["pairs"] / med(times), 2), loss1=losses[0], lossLast=losses[-1], accs=[round(a, 2) for a in accs],
                 stepTimes=[round(t, 2) for t in times], gradScalerFinal=scaler.get_scale(), ok=True)

    elif kind == "dpo":
        tok = load_tok(name)
        data = prefs(tok, job["steps"] * job["pairs"], T)
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, device_map=DEV)
        R["loadedClass"] = type(model).__name__; R["loadSec"] = round(time.time() - t0, 1)
        ref = None
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        model.config.use_cache = False
        if job["mode"] == "lora":
            model = lora_model(model)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        else:
            import bitsandbytes as bnb
            ref = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map=DEV).eval()
            for p in ref.parameters(): p.requires_grad = False
            opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=5e-7)
        tr = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in tr)
        model.train(); scaler = torch.amp.GradScaler("cuda", init_scale=2048)
        torch.cuda.reset_peak_memory_stats(); times = []; refT = []; losses = []; margins = []; beta = 0.1
        for s in range(job["steps"]):
            chunk = data[s * job["pairs"]:(s + 1) * job["pairs"]]
            opt.zero_grad(set_to_none=True); t1 = time.time(); tl = 0; tref = 0; mg = 0
            for (p, c, r) in chunk:  # one pair per micro-batch (2 rows)
                ids, att = pad_batch([p + c, p + r], tok.pad_token_id, len(p) + max(len(c), len(r))); ids = ids.cuda(); att = att.cuda()
                plen = torch.tensor([len(p)] * 2, device=DEV)
                def refpass():
                    with torch.no_grad():
                        if ref is not None:
                            return seq_logp(ref, ids, att, plen)[0]
                        with model.disable_adapter():
                            model.eval(); x = seq_logp(model, ids, att, plen)[0]; model.train(); return x
                rl, dtt = timed(refpass); tref += dtt
                pl, _ = seq_logp(model, ids, att, plen)
                z = beta * ((pl[0] - rl[0]) - (pl[1] - rl[1]))
                loss = -torch.nn.functional.logsigmoid(z) / len(chunk)
                scaler.scale(loss).backward(); tl += loss.item(); mg += z.item() / beta / len(chunk)
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); times.append(time.time() - t1); refT.append(tref); losses.append(tl); margins.append(mg)
            print(f"step {s+1} loss {tl:.4f} margin {mg:.3f} {times[-1]:.2f}s ref {tref:.2f}s", flush=True)
        R.update(stepsDone=len(times), pairsPerStep=job["pairs"], seq=T, sPerStepMedian=med(times), refForwardSecMedian=med(refT), peakMemGB=peak(),
                 loss1=losses[0], lossLast=losses[-1], marginLast=margins[-1], stepTimes=[round(t, 2) for t in times], ok=True,
                 refMode="separate fp16 copy" if ref is not None else "adapter disabled (no extra copy)")

    elif kind == "toolsft":
        from datasets import load_dataset
        tok = load_tok(name)
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in ds["text"][:20000] if len(t.strip()) > 40)[:900000]
        ids_all = tok(text, return_tensors="pt").input_ids[0]
        B = job["batch"]; N = job["steps"]
        stream = ids_all[: B * T * N].view(N, B, T)
        t0 = time.time()
        if job["mode"] == "qlora":
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, dtype=torch.float16, device_map=DEV)
        else:
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map=DEV)
        R["loadedClass"] = type(model).__name__; R["loadSec"] = round(time.time() - t0, 1)
        for p in model.parameters(): p.requires_grad = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        model.config.use_cache = False
        model = lora_model(model)
        tr = [p for p in model.parameters() if p.requires_grad]
        R["trainableParams"] = sum(p.numel() for p in tr)
        opt = torch.optim.AdamW(tr, lr=2e-4); model.train(); scaler = torch.amp.GradScaler("cuda", init_scale=2048)
        torch.cuda.reset_peak_memory_stats(); times = []; losses = []; micro = job.get("micro", B); nonf = 0
        for s in range(N):
            t1 = time.time()
            while True:
                try:
                    opt.zero_grad(set_to_none=True); tl = 0
                    for i in range(0, B, micro):
                        x = stream[s][i:i + micro].cuda()
                        with torch.autocast("cuda", dtype=torch.float16):
                            loss = model(input_ids=x, labels=x).loss
                        frac = x.shape[0] / B; tl += loss.item() * frac; scaler.scale(loss * frac).backward()
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
                    break
                except torch.OutOfMemoryError:
                    R.setdefault("oomRetries", []).append({"step": s, "micro": micro}); opt.zero_grad(set_to_none=True); free()
                    if micro == 1: raise
                    micro //= 2
            torch.cuda.synchronize(); times.append(time.time() - t1); losses.append(tl)
            nonf += int(not (tl == tl and abs(tl) != float("inf")))
            print(f"step {s+1} loss {tl:.4f} {times[-1]:.2f}s", flush=True)
        R.update(stepsDone=len(times), batch=B, seq=T, microBatch=micro, sPerStepMedian=med(times), tokensPerSec=round(B * T / med(times), 1), peakMemGB=peak(),
                 loss1=losses[0], lossLast=losses[-1], nonFinite=nonf, stepTimes=[round(t, 2) for t in times], ok=True)

    elif kind == "grpo":
        from datasets import load_dataset
        tok = load_tok(name); tok.padding_side = "left"
        ds = load_dataset("openai/gsm8k", "main", split="train[:64]")
        prompts = [f"Question: {q}\nSolve step by step, then give the final answer after '####'.\nAnswer:" for q in ds["question"]]
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map=DEV)
        R["loadedClass"] = type(model).__name__; R["loadSec"] = round(time.time() - t0, 1)
        for p in model.parameters(): p.requires_grad = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
        model = lora_model(model)
        tr = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(tr, lr=1e-5); scaler = torch.amp.GradScaler("cuda", init_scale=2048)
        R["trainableParams"] = sum(p.numel() for p in tr)
        NEW = job.get("new", 256); rows = []
        torch.cuda.reset_peak_memory_stats()
        for it in range(job["iters"]):
            for (nprompt, G) in job["shapes"]:
                batch_prompts = [pr for pr in prompts[it * nprompt:(it + 1) * nprompt] for _ in range(G)]
                enc = tok(batch_prompts, return_tensors="pt", padding=True).to(DEV)
                model.eval(); model.config.use_cache = True; torch.cuda.reset_peak_memory_stats()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    out, tg = timed(lambda: model.generate(**enc, max_new_tokens=NEW, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id))
                gen = out[:, enc.input_ids.shape[1]:]
                eos = tok.eos_token_id
                lens = []
                for row in gen.tolist():
                    n = len(row)
                    for j, t in enumerate(row):
                        if t == eos or t == tok.pad_token_id: n = j + 1; break
                    lens.append(n)
                genpeak = peak()
                # policy-gradient update on these samples (random advantages: only timing and memory are measured)
                model.train(); model.config.use_cache = False; torch.cuda.reset_peak_memory_stats()
                adv = torch.randn(len(lens), device=DEV)
                full_att = torch.cat([enc.attention_mask, torch.ones_like(gen)], 1)
                def upd():
                    opt.zero_grad(set_to_none=True); mb = job.get("micro", 4)
                    for i in range(0, out.shape[0], mb):
                        o = out[i:i + mb]; a = full_att[i:i + mb]
                        with torch.autocast("cuda", dtype=torch.float16):
                            lg = model(input_ids=o, attention_mask=a).logits[:, enc.input_ids.shape[1] - 1:-1]
                        lp = torch.log_softmax(lg.float(), -1).gather(-1, gen[i:i + mb].unsqueeze(-1)).squeeze(-1)
                        m = torch.zeros_like(lp)
                        for k, n in enumerate(lens[i:i + mb]): m[k, :n] = 1
                        loss = -((lp * adv[i:i + mb].unsqueeze(1) * m).sum() / m.sum()) * (o.shape[0] / out.shape[0])
                        scaler.scale(loss).backward()
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
                _, tu = timed(upd)
                rows.append(dict(iter=it, prompts=nprompt, G=G, completions=len(lens), promptTokens=int(enc.input_ids.shape[1]), genSec=round(tg, 2),
                                 genTokens=int(sum(lens)), genTokPerSec=round(sum(lens) / tg, 1), meanLen=round(sum(lens) / len(lens), 1),
                                 hitCap=int(sum(1 for n in lens if n >= NEW)), updSec=round(tu, 2), genPeakGB=genpeak, updPeakGB=peak()))
                print(rows[-1], flush=True); R["rows"] = rows; save()
                free()
        R.update(ok=True)
        R["sample"] = tok.decode(gen[0][:lens[0]], skip_special_tokens=True)[:300]

    elif kind in ("kd", "kd_onpolicy"):
        from datasets import load_dataset
        tname = job["teacher"]; sname = name
        tok = load_tok(sname)
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:400]")
        texts = [e["prompt"] + "\n\n" + e["chosen"][-1]["content"] for e in ds]
        t0 = time.time()
        if job["teacher_quant"]:
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            teacher = AutoModelForCausalLM.from_pretrained(tname, quantization_config=q, dtype=torch.float16, device_map=DEV)
        else:
            teacher = AutoModelForCausalLM.from_pretrained(tname, dtype=torch.float16, device_map=DEV)
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False
        R["teacherLoadSec"] = round(time.time() - t0, 1); R["teacherGB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        R["teacherVocab"] = teacher.config.get_text_config().vocab_size
        t0 = time.time()
        if job["student_mode"] == "full":
            student = AutoModelForCausalLM.from_pretrained(sname, dtype=torch.float32, device_map=DEV)
            import bitsandbytes as bnb
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); student.enable_input_require_grads()
            opt = bnb.optim.AdamW8bit([p for p in student.parameters() if p.requires_grad], lr=1e-5)
        else:
            student = AutoModelForCausalLM.from_pretrained(sname, dtype=torch.float16, device_map=DEV)
            for p in student.parameters(): p.requires_grad = False
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); student.enable_input_require_grads()
            student = lora_model(student)
            opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
        student.config.use_cache = False
        tr = [p for p in student.parameters() if p.requires_grad]
        R["studentLoadSec"] = round(time.time() - t0, 1); R["studentVocab"] = student.get_base_model().config.get_text_config().vocab_size if hasattr(student, "get_base_model") else student.config.get_text_config().vocab_size
        R["trainableParams"] = sum(p.numel() for p in tr); R["residentGBBeforeStep"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        scaler = torch.amp.GradScaler("cuda", init_scale=2048); torch.cuda.reset_peak_memory_stats()
        B = job["batch"]; rows = []; times = []
        tok.padding_side = "right"
        for s in range(job["steps"]):
            t1 = time.time(); rec = {}
            if kind == "kd":
                enc = tok(texts[s * B:(s + 1) * B], return_tensors="pt", padding="max_length", truncation=True, max_length=T).to(DEV)
                ids, att = enc.input_ids, enc.attention_mask
                micro = job.get("micro", B); opt.zero_grad(set_to_none=True); tl = 0
                for i in range(0, B, micro):
                    x = ids[i:i + micro]; a = att[i:i + micro]
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        tl_logits = teacher(input_ids=x, attention_mask=a).logits.float()
                    with torch.autocast("cuda", dtype=torch.float16):
                        sl = student(input_ids=x, attention_mask=a).logits.float()
                    m = a[:, 1:].float()
                    # forward KL(teacher || student) per token
                    tp = torch.log_softmax(tl_logits[:, :-1], -1); sp = torch.log_softmax(sl[:, :-1], -1)
                    kl = (tp.exp() * (tp - sp)).sum(-1)
                    loss = (kl * m).sum() / m.sum() * (x.shape[0] / B)
                    scaler.scale(loss).backward(); tl += loss.item()
                    del tl_logits, sl, tp, sp, kl
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
                rec = dict(loss=tl)
            else:
                tok.padding_side = "left"
                pr = [e["prompt"][:400] for e in ds.select(range(s * B, (s + 1) * B))]
                enc = tok(pr, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEV)
                student.eval(); student.config.use_cache = True
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    out, tg = timed(lambda: student.generate(**enc, max_new_tokens=job["new"], do_sample=True, temperature=1.0, top_k=0, pad_token_id=tok.pad_token_id))
                student.train(); student.config.use_cache = False
                P = enc.input_ids.shape[1]; gen = out[:, P:]
                att = torch.cat([enc.attention_mask, torch.ones_like(gen)], 1)
                def tscore():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        lg = teacher(input_ids=out, attention_mask=att).logits[:, P - 1:-1].float()
                        return torch.log_softmax(lg, -1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
                tlp, tt = timed(tscore)
                def sup():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16):
                        lg = student(input_ids=out, attention_mask=att).logits[:, P - 1:-1]
                    slp = torch.log_softmax(lg.float(), -1).gather(-1, gen.unsqueeze(-1)).squeeze(-1)
                    # per-token reverse-KL "reward" (teacher logp - student logp) as the advantage, REINFORCE-style
                    adv = (tlp - slp).detach()
                    loss = -(slp * adv).mean()
                    scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(tr, 1.0); scaler.step(opt); scaler.update()
                    return float(adv.mean())
                am, tu = timed(sup)
                rec = dict(genSec=round(tg, 2), teacherScoreSec=round(tt, 2), updSec=round(tu, 2), meanRevKLEstimate=-am, genTokens=int(gen.numel()))
            torch.cuda.synchronize(); times.append(time.time() - t1); rec["stepSec"] = round(times[-1], 2); rows.append(rec); print(rec, flush=True)
        R.update(rows=rows, batch=B, seq=T, sPerStepMedian=med(times, 1), peakMemGB=peak(), ok=True)

    elif kind == "gold_rm":
        from transformers import AutoModelForSequenceClassification
        tok = load_tok(name)
        data = prefs(tok, 64, T)
        t0 = time.time()
        if job.get("quant"):
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, quantization_config=q, dtype=torch.float16, device_map=DEV)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, dtype=torch.float16, device_map=DEV)
        model.config.pad_token_id = tok.pad_token_id
        try: model.config.get_text_config().pad_token_id = tok.pad_token_id
        except Exception: pass
        model.eval(); R["loadSec"] = round(time.time() - t0, 1); R["loadedClass"] = type(model).__name__; R["residentGB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        torch.cuda.reset_peak_memory_stats(); correct = 0; times = []
        for i in range(0, len(data), 8):
            sub = data[i:i + 8]
            seqs = [p + c for p, c, r in sub] + [p + r for p, c, r in sub]
            ids, att = pad_batch(seqs, tok.pad_token_id, T)
            ids = ids.cuda(); att = att.cuda()
            def f():
                with torch.no_grad():
                    return model(input_ids=ids, attention_mask=att).logits.squeeze(-1).float()
            rew, dt = timed(f); times.append(dt)
            correct += int((rew[:len(sub)] > rew[len(sub):]).sum().item())
        R.update(pairs=len(data), pairAccuracyOnUltraFeedbackTrain=round(correct / len(data), 3), secPerBatch16Seq=med(times, 1), seqPerSec=round(16 / med(times, 1), 1), peakMemGB=peak(), ok=True)

    elif kind == "sample_n":
        from datasets import load_dataset
        tok = load_tok(name); tok.padding_side = "left"
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:16]")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=(torch.float32 if job.get("fp32") else torch.float16), device_map=DEV).eval()
        n = job["n"]; rows = []; R["dtype"] = "float32" if job.get("fp32") else "float16"
        for i in range(3):
            prompts = [ds[j]["prompt"][:400] for j in range(i * 4, i * 4 + 4) for _ in range(n)]
            msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
            enc = tok(msgs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
            torch.cuda.reset_peak_memory_stats()
            out, dt = timed(lambda: model.generate(**enc, max_new_tokens=job["new"], do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id))
            gen = out[:, enc.input_ids.shape[1]:]
            rows.append(dict(prompts=4, n=n, completions=len(prompts), sec=round(dt, 2), meanNewTokens=round(float((gen != tok.pad_token_id).sum(1).float().mean()), 1), peakGB=peak()))
            print(rows[-1], flush=True)
        R.update(rows=rows, ok=True)

    elif kind.startswith("trl_"):
        import trl
        from datasets import load_dataset
        from peft import LoraConfig
        R["trlVersion"] = trl.__version__
        tok = load_tok(name)
        lc = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM" if kind != "trl_rm" else "SEQ_CLS")
        mk = dict(dtype=torch.float32)
        t0 = time.time()
        if kind == "trl_rm":
            from trl import RewardTrainer, RewardConfig
            ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:96]")
            cfg = RewardConfig(output_dir="/kaggle/working/o_rm", max_length=512, per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=job["steps"],
                               fp16=True, gradient_checkpointing=True, report_to="none", logging_steps=1, learning_rate=1e-4, save_strategy="no", model_init_kwargs=mk)
            trainer = RewardTrainer(model=name, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lc)
        elif kind == "trl_dpo":
            from trl import DPOTrainer, DPOConfig
            ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:96]")
            cfg = DPOConfig(output_dir="/kaggle/working/o_dpo", max_length=512, per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=job["steps"],
                            fp16=True, gradient_checkpointing=True, report_to="none", logging_steps=1, learning_rate=5e-6, save_strategy="no", model_init_kwargs=mk)
            trainer = DPOTrainer(model=name, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lc)
        elif kind == "trl_grpo":
            from trl import GRPOTrainer, GRPOConfig
            g = load_dataset("openai/gsm8k", "main", split="train[:64]")
            ds = g.map(lambda e: {"prompt": f"Question: {e['question']}\nSolve step by step, then give the final answer after '####'.\nAnswer:", "gt": e["answer"].split("####")[-1].strip()})
            def reward(completions, gt, **kw):
                return [1.0 if str(a) in c else 0.0 for c, a in zip(completions, gt)]
            cfg = GRPOConfig(output_dir="/kaggle/working/o_grpo", num_generations=4, per_device_train_batch_size=4, gradient_accumulation_steps=1, max_completion_length=job.get("new", 128),
                             max_steps=job["steps"], fp16=True, gradient_checkpointing=True, report_to="none", logging_steps=1, learning_rate=1e-5, save_strategy="no", model_init_kwargs=mk,
                             use_vllm=False, beta=0.0)
            trainer = GRPOTrainer(model=name, reward_funcs=reward, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lc)
        elif kind == "trl_grpo_tools":
            from trl import GRPOTrainer, GRPOConfig
            from datasets import Dataset
            import random
            random.seed(0)
            def multiply(a: int, b: int) -> int:
                """Multiply two integers.

                Args:
                    a: First integer.
                    b: Second integer.

                Returns:
                    The product of a and b.
                """
                return a * b
            rows = []
            for _ in range(32):
                a, b = random.randint(11, 99), random.randint(11, 99)
                rows.append({"prompt": [{"role": "user", "content": f"What is {a} times {b}? Use the multiply tool."}], "gt": str(a * b)})
            ds = Dataset.from_list(rows)
            def reward(completions, gt, **kw):
                out = []
                for c, a in zip(completions, gt):
                    txt = c[-1]["content"] if isinstance(c, list) and c else str(c)
                    out.append(1.0 if a in str(txt) else 0.0)
                return out
            cfg = GRPOConfig(output_dir="/kaggle/working/o_grpot", num_generations=4, per_device_train_batch_size=4, gradient_accumulation_steps=1, max_completion_length=job.get("new", 256),
                             max_steps=job["steps"], fp16=True, gradient_checkpointing=True, report_to="none", logging_steps=1, learning_rate=1e-5, save_strategy="no", model_init_kwargs=mk,
                             use_vllm=False, beta=0.0, max_tool_calling_iterations=3)
            trainer = GRPOTrainer(model=name, reward_funcs=reward, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lc, tools=[multiply])
        elif kind == "trl_variant":
            v = job["variant"]
            common = dict(max_length=512, per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=job["steps"], fp16=True, gradient_checkpointing=True,
                          report_to="none", logging_steps=1, learning_rate=5e-6, save_strategy="no", model_init_kwargs=mk)
            if v == "kto":
                from trl import KTOTrainer, KTOConfig
                ds = load_dataset("trl-lib/kto-mix-14k", split="train[:96]")
                trainer = KTOTrainer(model=name, args=KTOConfig(output_dir="/kaggle/working/o_kto", **common), train_dataset=ds, processing_class=tok, peft_config=lc)
            elif v == "orpo":
                from trl.experimental.orpo import ORPOTrainer, ORPOConfig
                ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:96]")
                trainer = ORPOTrainer(model=name, args=ORPOConfig(output_dir="/kaggle/working/o_orpo", **common), train_dataset=ds, processing_class=tok, peft_config=lc)
            elif v == "simpo":
                from trl.experimental.cpo import CPOTrainer, CPOConfig
                ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:96]")
                trainer = CPOTrainer(model=name, args=CPOConfig(output_dir="/kaggle/working/o_simpo", loss_type="simpo", cpo_alpha=0.0, **common), train_dataset=ds, processing_class=tok, peft_config=lc)
            elif v == "ipo":
                from trl import DPOTrainer, DPOConfig
                ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:96]")
                trainer = DPOTrainer(model=name, args=DPOConfig(output_dir="/kaggle/working/o_ipo", loss_type="ipo", **common), train_dataset=ds, processing_class=tok, peft_config=lc)
        elif kind == "trl_distill":
            from trl import DistillationTrainer, DistillationConfig
            ds = load_dataset("trl-lib/tldr", split="train[:64]") if False else load_dataset("openai/gsm8k", "main", split="train[:64]").map(lambda e: {"prompt": f"Question: {e['question']}\nAnswer:"}, remove_columns=["question", "answer"])
            cfg = DistillationConfig(output_dir="/kaggle/working/o_dist", per_device_train_batch_size=4, gradient_accumulation_steps=1, max_completion_length=job.get("new", 128),
                                     max_steps=job["steps"], fp16=True, gradient_checkpointing=True, report_to="none", logging_steps=1, learning_rate=1e-5, save_strategy="no",
                                     model_init_kwargs=dict(dtype="float32"), teacher_model_init_kwargs=dict(dtype="float16"), beta=1.0)
            trainer = DistillationTrainer(model=name, teacher_model=job["teacher"], args=cfg, train_dataset=ds, processing_class=tok, peft_config=lc)
        R["initSec"] = round(time.time() - t0, 1)
        # trap check from batch B: TRL may recast LoRA weights to bf16
        dts = sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}); R["trainableDtypesAfterInit"] = dts
        for p in trainer.model.parameters():
            if p.requires_grad and p.dtype != torch.float32: p.data = p.data.float()
        torch.cuda.reset_peak_memory_stats(); t1 = time.time()
        trainer.train()
        torch.cuda.synchronize(); R["trainWallSec"] = round(time.time() - t1, 1); R["sPerStepMean"] = round(R["trainWallSec"] / job["steps"], 2); R["peakMemGB"] = peak()
        R["log"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} for h in trainer.state.log_history][-job["steps"] - 1:]
        R["ok"] = True
    R["warnings"] = logbuf.getvalue()[:1500]
except Exception as e:
    R["ok"] = False; R["error"] = f"{type(e).__name__}: {str(e)[:900]}"; R["trace"] = traceback.format_exc()[-2200:]
    R["warnings"] = logbuf.getvalue()[:1500]
    try: R["peakMemGB"] = peak()
    except Exception: pass
save()
