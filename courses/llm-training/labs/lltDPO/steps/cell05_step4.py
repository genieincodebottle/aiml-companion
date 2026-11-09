from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
GPU = torch.device("cuda")
RUN_SFT = False          # True trains the SFT model on your T4; False downloads ours from the Hub
runs = {}                # every measured number this notebook prints is also kept here

def lora_on(model, r=16):
    # LoRA on every linear layer of a frozen fp16 model; the adapter weights stay fp32 for the GradScaler
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    m = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for q in m.parameters():
        if q.requires_grad:
            q.data = q.data.float()
    return m

def plan(n, size, seed, lengths):
    # shuffle, sort each chunk of 64 by length, cut into batches, shuffle the batches: less padding, same data
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    out = []
    for k in range(0, n, 64):
        chunk = sorted(order[k:k + 64], key=lambda i: -lengths[i])
        out += [chunk[j:j + size] for j in range(0, len(chunk), size)]
    rng.shuffle(out)
    return out

def cosine(step, total, warm):
    return (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

def train_sft(pairs, lr=2e-4, size=8, micro=4, seed=0):
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(GPU)
    model = lora_on(base)
    steps = plan(len(pairs), size, seed, [len(x["c"]) for x in pairs])
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, len(steps), max(1, len(steps) // 20)))
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    torch.cuda.reset_peak_memory_stats()
    t0, hist = time.time(), []
    model.train()
    for step, idx in enumerate(steps):
        opt.zero_grad(set_to_none=True)
        reply_tokens = sum(len(pairs[i]["c"]) - len(pairs[i]["p"]) for i in idx)
        total = 0.0
        for k in range(0, len(idx), micro):
            part = [pairs[i] for i in idx[k:k + micro]]
            with torch.autocast("cuda", dtype=torch.float16):
                lp = seq_logps(model, [x["c"] for x in part], [len(x["p"]) for x in part])
            loss = -lp.sum() / reply_tokens          # mean NLL per reply token over the whole batch
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        hist.append(total)
        if (step + 1) % 75 == 0 or step == 0:
            print("SFT step", step + 1, "of", len(steps), "| loss", round(sum(hist[-75:]) / len(hist[-75:]), 4), "| seconds", round(time.time() - t0))
    runs["sft"] = {"steps": len(steps), "sec": round(time.time() - t0, 1), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "lossFirst75": round(sum(hist[:75]) / 75, 4), "lossLast75": round(sum(hist[-75:]) / 75, 4),
                   "loss": [round(x, 4) for x in hist]}
    print({k: v for k, v in runs["sft"].items() if k != "loss"})
    merged = model.merge_and_unload()            # fold the adapter into the fp16 weights
    merged.config.use_cache = True
    merged.save_pretrained("sft-model")
    tok.save_pretrained("sft-model")
    del model, merged, base
    gc.collect()
    torch.cuda.empty_cache()
    return "sft-model"

if RUN_SFT:
    SFT_DIR = train_sft(sft_pairs)
else:
    SFT_DIR = os.path.join(snapshot_download("aimlcompanion/llm-training-lab", allow_patterns="lltDPO/sft-s0/*"), "lltDPO", "sft-s0")

def load_sft():
    m = AutoModelForCausalLM.from_pretrained(SFT_DIR, dtype=torch.float16).to(GPU)
    m.config.use_cache = True
    return m

def generate(model, prompts, temperature=0.7, max_new=256, seed=0, batch=50):
    # one sampled reply per prompt, left padding; a reply ends at its first <|endoftext|>
    tok.padding_side, out = "left", []
    model.eval()
    for k in range(0, len(prompts), batch):
        enc = tok([TEMPLATE.format(q) for q in prompts[k:k + batch]], return_tensors="pt", padding=True).to(GPU)
        torch.manual_seed(seed + k)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                               max_new_tokens=max_new, use_cache=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:].tolist():
            ended = tok.eos_token_id in row
            n = row.index(tok.eos_token_id) + 1 if ended else len(row)
            out.append({"text": tok.decode(row[:n], skip_special_tokens=True).strip(), "tokens": n, "ended": ended})
    tok.padding_side = "right"
    return out

sft = load_sft()
print(generate(sft, ["Name three fruits that grow in India."], temperature=0.7)[0])
