import math, random, time
GPU = torch.device("cuda")
RUN_TRAINING = True       # False loads our seed-0 reward model from the Hub instead (about 1 min)

def pad(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(GPU), mask.to(GPU)

def batch_plan(pairs, size, seed):
    # shuffle, sort each chunk of 64 pairs by length, cut into batches, shuffle the batches: less padding, same data
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    plan = []
    for k in range(0, len(order), 64):
        chunk = sorted(order[k:k + 64], key=lambda i: -max(len(pairs[i]["c"]), len(pairs[i]["r"])))
        plan += [chunk[j:j + size] for j in range(0, len(chunk), size)]
    rng.shuffle(plan)
    return plan

def train_rm(pairs, seed=0, lr=1e-5, size=8):
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, dtype=torch.float32).to(GPU)
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    plan = batch_plan(pairs, size, seed)
    warm = max(1, int(0.03 * len(plan)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / (len(plan) - warm)))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2048)    # fp16 gradients need loss scaling on a T4
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, hist = time.time(), []
    for step, idx in enumerate(plan):
        chunk = [pairs[i] for i in idx]
        ids, mask = pad([p["c"] for p in chunk] + [p["r"] for p in chunk])
        with torch.autocast("cuda", dtype=torch.float16):
            r = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1).float()
        margin = r[:len(chunk)] - r[len(chunk):]
        loss = -torch.nn.functional.logsigmoid(margin).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        hist.append((loss.item(), (margin > 0).float().mean().item()))
        if (step + 1) % 75 == 0 or step == 0:
            recent = hist[-75:]
            print("step", step + 1, "of", len(plan), "| loss", round(sum(h[0] for h in recent) / len(recent), 4),
                  "| batch accuracy", round(sum(h[1] for h in recent) / len(recent), 3), "| seconds", round(time.time() - t0))
    print("peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return model.eval()

if RUN_TRAINING:
    rm = train_rm(train, seed=0)
else:
    rm = AutoModelForSequenceClassification.from_pretrained("aimlcompanion/llm-training-lab", subfolder="lltRewardModels/rm-s0",
                                                            dtype=torch.float32).to(GPU).eval()
