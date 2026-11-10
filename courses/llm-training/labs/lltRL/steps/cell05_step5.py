RUN_EXPERIMENT = False   # True trains two more policies, 31 and 35 min on our shared A100. False: read our runs below

def summary(name, r):
    e = r["evals"][-1]
    print(name, "| held-out score", e["score"], "| KL", e["kl"], "| distinct tokens", e["distinct"], "| last clip fraction", r["hist"][-1]["clipfrac"])
    print("   ", repr(e["samples"][0][:160]))

if RUN_EXPERIMENT:
    for kl in [0.0, 0.2]:
        _, runs["ppo-kl" + str(kl) + "-s0"] = train_ppo(dict(PPO, kl_coef=kl), log_every=50)
        summary("kl_coef " + str(kl), runs["ppo-kl" + str(kl) + "-s0"])
        gc.collect(); torch.cuda.empty_cache()
