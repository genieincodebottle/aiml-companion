RUN_PPO = False      # False: download our trained policy and only evaluate it. True: train it here (27 min on a T4, 28 min on our shared A100)
PPO = dict(iters=200, batch=64, epochs=4, minibatch=16, lr=1e-5, kl_coef=0.05, clip=0.2, vf_clip=0.2, gamma=1.0, lam=0.95, seed=0)
AMP = DEV.type == "cuda"     # fp16 autocast on the GPU; the trained weights stay fp32

def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=AMP)

@torch.no_grad()
def evaluate(policy, ref, rm, n=128):
    # held-out prompts, one sample each with a fixed seed: mean score, mean KL to the reference, share of distinct tokens
    rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if DEV.type == "cuda" else None)
    seqs = generate(policy, eval_prompts[:n], seed=1234)
    torch.random.set_rng_state(rng[0])
    if rng[1] is not None:
        torch.cuda.set_rng_state(rng[1])     # put the training RNG back, so evaluating does not change the training samples
    with autocast():
        lp, _ = response_logps(policy, seqs)
        rlp, _ = response_logps(ref, seqs)
    resp = seqs[:, P_LEN:]
    distinct = sum(len(set(r.tolist())) for r in resp) / resp.numel()
    return dict(score=round(score(seqs, rm).mean().item(), 4), kl=round((lp - rlp).sum(1).mean().item(), 4),
                distinct=round(distinct, 4), samples=tok.batch_decode(seqs[:4], skip_special_tokens=True))

def train_ppo(cfg, log_every=10, eval_every=25):
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"])
    policy = AutoModelForCausalLM.from_pretrained(POLICY, dtype=torch.float32).to(DEV)
    value = ValueModel().to(DEV)
    opt_p = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"], eps=1e-5)
    opt_v = torch.optim.AdamW(value.parameters(), lr=cfg["lr"], eps=1e-5)
    sc_p = torch.amp.GradScaler("cuda", init_scale=2048, enabled=AMP)
    sc_v = torch.amp.GradScaler("cuda", init_scale=2048, enabled=AMP)
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist, evals, t0 = [], [], time.time()
    for it in range(cfg["iters"] + 1):
        if it % eval_every == 0 or it == cfg["iters"]:
            policy.eval()
            evals.append(dict(iter=it, **evaluate(policy, ref, rm)))
            print("eval", it, {k: v for k, v in evals[-1].items() if k != "samples"}, flush=True)
        if it == cfg["iters"]:
            break
        # 1. Rollout: sample responses, score them, record old log-probs, reference log-probs and values
        policy.eval()
        prompts = train_prompts[it * cfg["batch"]:(it + 1) * cfg["batch"]]
        seqs = generate(policy, prompts)
        with torch.no_grad(), autocast():
            sc = score(seqs, rm)
            old_lp, _ = response_logps(policy, seqs)
            ref_lp, _ = response_logps(ref, seqs)
            old_v = value(seqs)[:, P_LEN - 1:-1].float()     # value of the state before each response token
        # 2. Rewards and advantages
        rewards = shaped_rewards(sc, old_lp, ref_lp, cfg["kl_coef"])
        adv, ret = gae(rewards, old_v, cfg["gamma"], cfg["lam"])
        adv = whiten(adv)
        # 3. Several epochs of minibatch updates on this one batch
        policy.train(); stats = []
        for epoch in range(cfg["epochs"]):
            for mb in torch.randperm(len(seqs), device=DEV).split(cfg["minibatch"]):
                with autocast():
                    new_lp, ent = response_logps(policy, seqs[mb])
                    pl, cf = ppo_policy_loss(new_lp, old_lp[mb], adv[mb], cfg["clip"])
                opt_p.zero_grad(set_to_none=True)
                sc_p.scale(pl).backward(); sc_p.unscale_(opt_p)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); sc_p.step(opt_p); sc_p.update()
                with autocast():
                    vl = value_loss(value(seqs[mb])[:, P_LEN - 1:-1].float(), old_v[mb], ret[mb], cfg["vf_clip"])
                opt_v.zero_grad(set_to_none=True)
                sc_v.scale(vl).backward(); sc_v.unscale_(opt_v)
                torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0); sc_v.step(opt_v); sc_v.update()
                with torch.no_grad():
                    stats.append([cf.item(), kl_k3(new_lp.detach(), old_lp[mb]).mean().item(), vl.item(), ent.mean().item(),
                                  torch.exp(new_lp.detach() - old_lp[mb]).max().item()])
        s = torch.tensor(stats).mean(0).tolist()
        hist.append(dict(iter=it + 1, score=round(sc.mean().item(), 4), kl=round((old_lp - ref_lp).sum(1).mean().item(), 4),
                         clipfrac=round(s[0], 4), approxKL=round(s[1], 5), valueLoss=round(s[2], 4), entropy=round(s[3], 4),
                         ratioMax=round(max(r[4] for r in stats), 3), sec=round(time.time() - t0, 1)))
        if (it + 1) % log_every == 0:
            print(hist[-1], flush=True)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV.type == "cuda" else None
    del value, opt_p, opt_v
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return policy, dict(cfg=cfg, hist=hist, evals=evals, peakGB=peak, sec=round(time.time() - t0, 1))

runs = {}
if RUN_PPO:
    trained, runs["ppo-kl0.05-s0"] = train_ppo(PPO)
else:
    trained = AutoModelForCausalLM.from_pretrained("aimlcompanion/llm-training-lab", subfolder="lltRL/ppo-kl0.05-s0", dtype=torch.float32).to(DEV)
start, end = evaluate(policy, ref, rm), evaluate(trained, ref, rm)
print("held-out score", start["score"], "->", end["score"], "| KL to the reference", start["kl"], "->", end["kl"],
      "| distinct tokens", start["distinct"], "->", end["distinct"])
for t in end["samples"][:3]:
    print(repr(t))
