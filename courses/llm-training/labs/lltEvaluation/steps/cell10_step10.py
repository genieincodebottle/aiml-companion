from peft import LoraConfig, get_peft_model

SEED = int(os.environ.get("LLT_SEED", "0"))
SFT_STEPS = 3 if SMOKE else 250

def dolly_rows(seed, n=2000):
    # Benign instruction data: Dolly 15k, written by Databricks employees, CC-BY-SA-3.0.
    rows = dolly.shuffle(seed=100 + seed).select(range(n))
    return [{"prompt": r["instruction"] + (NL + NL + r["context"] if r["context"] else ""), "response": r["response"]} for r in rows]

def encode_sft(tok, rows, max_len=512):
    # Chat-formatted prompt with no loss on it, then the answer and the end-of-turn token with loss.
    out = []
    for r in rows:
        p = tok(chat(tok, [{"role": "user", "content": r["prompt"]}]), add_special_tokens=False)["input_ids"]
        a = tok(r["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        out.append((ids, ([-100] * len(p) + a)[:max_len]))
    return out

def train_lora(model, tok, rows, seed, steps=SFT_STEPS, lr=1e-4, micro=4, accum=2):
    # LoRA r16 on every linear layer; frozen fp16 base, fp32 adapters, fp16 autocast with a gradient scaler.
    torch.manual_seed(seed)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    pm.gradient_checkpointing_enable()
    pm.enable_input_require_grads()
    data = encode_sft(tok, rows)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler(device, init_scale=2048, enabled=device == "cuda")
    rng = np.random.default_rng(seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pm.train()
    log, t = [], time.time()
    for step in range(steps):
        for _ in range(accum):
            batch = [data[i] for i in rng.choice(len(data), size=micro, replace=False)]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((micro, L), pad)
            lab = torch.full((micro, L), -100)
            att = torch.zeros((micro, L), dtype=torch.long)
            for r, (x, y) in enumerate(batch):
                ids[r, :len(x)], lab[r, :len(y)], att[r, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                loss = pm(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss / accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append(round(loss.item() * accum, 4))
        if step % 25 == 0 or step == steps - 1:
            print("step", step, "| loss", log[-1], "|", round(time.time() - t), "s", flush=True)
    pm.eval()
    pm.gradient_checkpointing_disable()
    return pm, {"loss": log, "seconds": round(time.time() - t, 1), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}

tuned_tok = LMS[TUNED].tokenizer
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
FT, FT_LOG = train_lora(LMS[TUNED].model, tuned_tok, dolly_rows(SEED), SEED)
print("trainable parameters", sum(p.numel() for p in FT.parameters() if p.requires_grad), "| peak GB", FT_LOG["peakGB"],
      "| loss", FT_LOG["loss"][0], "->", round(float(np.mean(FT_LOG["loss"][-25:])), 3), "(mean of the last 25 steps)")
