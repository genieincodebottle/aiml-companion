import math, os
from peft import LoraConfig, get_peft_model

STEPS, BATCH, MICRO, LR = 64, 32, 4, 2e-4      # 64 steps of 32 questions: each arm sees the same 2,048 training questions

def new_student():
    # frozen fp16 Qwen3-0.6B-Base with fp32 LoRA adapters, rank 16, on every linear layer
    m = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16).to(GPU)
    for p in m.parameters():
        p.requires_grad = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"))
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return m

def sync(model, copy):
    # write base + LoRA into the plain fp16 student, which then samples and evaluates without adapter layers
    with torch.no_grad():
        target = dict(copy.named_modules())
        for name, mod in model.get_base_model().named_modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                delta = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) * mod.scaling["default"]
                target[name].weight.copy_((mod.base_layer.weight.float() + delta).half())

def token_loss(kind, s_logits, t_logits, targets):
    lps = s_logits.log_softmax(-1)
    if kind == "sft":                                   # cross-entropy on the token that was written
        return -lps.gather(1, targets[:, None]).squeeze(1)
    lpt = t_logits.log_softmax(-1)
    if kind == "forward":                               # KL(teacher || student) over the whole vocabulary
        return (lpt.exp() * (lpt - lps)).sum(-1)
    return (lps.exp() * (lps - lpt)).sum(-1)            # "reverse": KL(student || teacher)

def train_student(kind, source, seed=0, steps=STEPS):
    # source "reference" or "teacher": fixed answers (off-policy). "student": answers sampled from the student every step (on-policy)
    torch.manual_seed(seed)
    model = new_student()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    warm = 3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    torch.cuda.reset_peak_memory_stats()
    losses, t0, sample_sec = [], time.time(), 0.0
    for step in range(steps):
        idx = order[step * BATCH:(step + 1) * BATCH]
        P = [train_prompt_ids[i] for i in idx]
        if source == "student":
            sync(model, student)
            g0 = time.time()
            C = generate(student, [prompt(train[i]["q"]) for i in idx], sample=True, batch=BATCH)
            sample_sec += time.time() - g0
        else:
            C = [(teacher_train if source == "teacher" else ref_train)[i] for i in idx]
        n_tok = sum(len(c) for c in C)
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for k in range(0, len(P), MICRO):
            ids, att, cm = pack(P[k:k + MICRO], C[k:k + MICRO])
            s = logits_at(model, ids, att, cm)
            t = None
            if kind != "sft":
                with torch.no_grad():
                    t = logits_at(teacher, ids, att, cm)
            loss = token_loss(kind, s, t, ids[:, 1:][cm[:, 1:].bool()]).sum() / n_tok
            scaler.scale(loss).backward()
            total += loss.item()
            del s, t, loss
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(round(total, 4))
        if step % 16 == 0 or step == steps - 1:
            print("step", step + 1, "of", steps, "| loss", round(total, 4), "| answer tokens", n_tok, "| seconds", round(time.time() - t0))
    stats = dict(trainSec=round(time.time() - t0, 1), sampleSec=round(sample_sec, 1),
                 peakGB=round(torch.cuda.max_memory_allocated() / 1e9, 2), loss=losses)
    return model, stats

def run_arm(name, kind, source, seed=0):
    # train one arm, save its adapter, evaluate it on the test questions, then free the GPU
    model, stats = train_student(kind, source, seed)
    model.save_pretrained(os.path.join("adapters", re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")))
    sync(model, student)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(student, name)
    R[name].update(stats)
    json.dump(R, open("distill_results.json", "w"))

run_arm("SFT on teacher answers", "sft", "teacher")