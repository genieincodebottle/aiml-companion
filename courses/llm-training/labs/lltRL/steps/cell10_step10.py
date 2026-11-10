START_FROM = "lltRL/grpo-s0-step130"   # our adapter after 130 of 150 steps; you train the last 20. None: train all 150 yourself
GRPO = dict(steps=150, prompts=4, G=8, lr=2e-5, beta=0.0, eps=0.2, eps_high=None, loss_type="dapo", adv="grpo", minibatches=2, micro=4,
            seed=0, eval_every=0, save_at=())

def train_grpo(cfg, model=None, first_step=0, tag="grpo", resume=None):
    # resume: a folder for a resume file, written every 10 steps; a dropped session restarts from it
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    model = model or load_policy()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=2048, enabled=AMP)
    n = cfg["prompts"] * cfg["G"]
    hist, evals, groups, t0 = [], [], [], time.time()
    state_file = os.path.join(resume, tag + "-resume.pt") if resume else None
    st = None
    if state_file and os.path.exists(state_file):
        try:
            st = torch.load(state_file, weights_only=False)
        except RuntimeError as e:                                   # a save cut off by a dropped session
            print("resume file unreadable, starting from step", first_step, "|", str(e)[:80], flush=True)
    if st:
        for p, v in zip(params, st["params"]):
            p.data.copy_(v)
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        torch.random.set_rng_state(st["rng"][0])
        if st["rng"][1] is not None:
            torch.cuda.set_rng_state(st["rng"][1])
        first_step, hist, evals, groups = st["step"], st["hist"], st["evals"], st["groups"]
        print("resumed at step", first_step, flush=True)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(first_step, cfg["steps"]):
        ts = time.time()
        batch = train[step * cfg["prompts"]:(step + 1) * cfg["prompts"]]
        texts = [prompt(x["q"]) for x in batch for _ in range(cfg["G"])]
        # 1. Sample G replies per prompt and score them
        rows = sample(model, texts, batch=n)
        gen_sec = time.time() - ts
        rewards = torch.tensor([correct(qtok.decode(r), x["gold"]) for r, x in zip(rows, [x for x in batch for _ in range(cfg["G"])])])
        if cfg["adv"] == "rloo":
            adv = rloo_advantages(rewards, cfg["G"])
        else:
            adv = group_advantages(rewards, cfg["G"], scale=cfg["adv"] == "grpo")
        adv = adv.to(DEV)
        # 2. One padded batch: prompt left-padded, reply right-padded, mask from each reply's length
        penc = qtok(texts, return_tensors="pt", padding=True).to(DEV)
        L = max(len(r) for r in rows)
        comp = torch.full((n, L), qtok.pad_token_id, device=DEV)
        cmask = torch.zeros((n, L), dtype=torch.long, device=DEV)
        for i, r in enumerate(rows):
            comp[i, :len(r)] = torch.tensor(r, device=DEV)
            cmask[i, :len(r)] = 1
        # 3. Log-probs of the sampling policy and of the reference (the base model: LoRA switched off)
        with torch.no_grad(), autocast():
            old_lp = torch.cat([completion_logps(model, penc.input_ids[i:i + cfg["micro"]], penc.attention_mask[i:i + cfg["micro"]],
                                                 comp[i:i + cfg["micro"]], cmask[i:i + cfg["micro"]]) for i in range(0, n, cfg["micro"])])
            with model.disable_adapter():
                ref_lp = torch.cat([completion_logps(model, penc.input_ids[i:i + cfg["micro"]], penc.attention_mask[i:i + cfg["micro"]],
                                                     comp[i:i + cfg["micro"]], cmask[i:i + cfg["micro"]]) for i in range(0, n, cfg["micro"])])
        # 4. Updates: the batch is split into minibatches, one optimizer step each, so later ones see a ratio that is not 1
        model.train(); clip, ent = [], []
        for mb in torch.randperm(n, device=DEV).chunk(cfg["minibatches"]):
            m = cmask[mb]
            denom = {"grpo": len(mb), "dr_grpo": len(mb) * MAX_NEW}.get(cfg["loss_type"], m.sum().item())
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(mb), cfg["micro"]):
                idx = mb[j:j + cfg["micro"]]
                with autocast():
                    new_lp = completion_logps(model, penc.input_ids[idx], penc.attention_mask[idx], comp[idx], cmask[idx])
                    loss = grpo_loss(new_lp, old_lp[idx], adv[idx], cmask[idx], cfg["eps"], cfg["eps_high"], cfg["beta"], ref_lp[idx],
                                     cfg["loss_type"], MAX_NEW, denom)
                scaler.scale(loss).backward()
                with torch.no_grad():
                    ratio = torch.exp(new_lp - old_lp[idx]); a = adv[idx][:, None]; hi = cfg["eps_high"] or cfg["eps"]
                    clipped = ((ratio > 1 + hi) & (a > 0)) | ((ratio < 1 - cfg["eps"]) & (a < 0))
                    clip.append(((clipped * cmask[idx]).sum() / cmask[idx].sum()).item())
                    ent.append((-(new_lp * cmask[idx]).sum() / cmask[idx].sum()).item())   # mean -log p of sampled tokens, an entropy estimate
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            scaler.step(opt); scaler.update()
        lens = [len(r) for r in rows]
        if step == first_step or (step + 1) % 50 == 0:     # keep two scored groups now and then, to read later
            groups.append(dict(step=step + 1, questions=[x["q"] for x in batch[:2]], gold=[x["gold"] for x in batch[:2]],
                               replies=[qtok.decode(r) for r in rows[:2 * cfg["G"]]], rewards=rewards[:2 * cfg["G"]].tolist(),
                               tokens=lens[:2 * cfg["G"]], advantages=[round(a, 4) for a in adv[:2 * cfg["G"]].tolist()]))
        hist.append(dict(step=step + 1, reward=round(rewards.mean().item(), 4), zeroStd=round((rewards.view(-1, cfg["G"]).std(1) == 0).float().mean().item(), 3),
                         meanLen=round(sum(lens) / n, 1), truncated=round(sum(1 for r in rows if len(r) >= MAX_NEW) / n, 3),
                         kl=round(((kl_k3(old_lp, ref_lp) * cmask).sum() / cmask.sum()).item(), 5), clipFrac=round(sum(clip) / len(clip), 4),
                         entropy=round(sum(ent) / len(ent), 4), gradNorm=round(gnorm, 3), genSec=round(gen_sec, 1), sec=round(time.time() - ts, 1)))
        if (step + 1) % 10 == 0 or step == first_step:
            print(hist[-1], flush=True)
        if cfg["eval_every"] and (step + 1) % cfg["eval_every"] == 0:
            evals.append(dict(step=step + 1, **evaluate_gsm(model)))
            print("eval", step + 1, {k: v for k, v in evals[-1].items() if k != "replies"}, flush=True)
        if step + 1 in cfg["save_at"]:
            model.save_pretrained(os.path.join(WORK, tag + "-step" + str(step + 1)))
        if state_file and (step + 1) % 10 == 0:
            torch.save(dict(params=[p.detach().cpu() for p in params], opt=opt.state_dict(), scaler=scaler.state_dict(),
                            rng=(torch.random.get_rng_state(), torch.cuda.get_rng_state() if DEV.type == "cuda" else None), step=step + 1, hist=hist, evals=evals, groups=groups), state_file + ".tmp")
            os.replace(state_file + ".tmp", state_file)            # write then rename, so a drop mid-save keeps the old file
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV.type == "cuda" else None
    return model, dict(cfg={k: v for k, v in cfg.items() if k != "save_at"}, firstStep=first_step, hist=hist, evals=evals, groups=groups, peakGB=peak,
                       sec=round(time.time() - t0, 1))

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
del policy; gc.collect()
if START_FROM:
    grpo_model, runs["grpo-last20"] = train_grpo(GRPO, load_policy(START_FROM), first_step=130)
else:
    grpo_model, runs["grpo-s0"] = train_grpo(GRPO)
h = runs.get("grpo-last20", runs.get("grpo-s0"))["hist"]
print("steps", h[0]["step"], "to", h[-1]["step"], "| mean reward", round(sum(x["reward"] for x in h) / len(h), 3),
      "| mean reply tokens", round(sum(x["meanLen"] for x in h) / len(h), 1), "| s per step", round(sum(x["sec"] for x in h) / len(h), 1))
