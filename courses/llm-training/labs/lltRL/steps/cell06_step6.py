RUN_EXPERIMENT = False   # True trains two more policies, 35 and 15 min on our shared A100. False: read our runs below

if RUN_EXPERIMENT:
    for name, cfg in [("noclip", dict(PPO, clip=1e6, vf_clip=1e6)),     # no clip on the policy or the value
                      ("epochs1", dict(PPO, epochs=1))]:                # one pass over each batch instead of four
        _, runs["ppo-" + name + "-s0"] = train_ppo(cfg, log_every=50)
        summary(name, runs["ppo-" + name + "-s0"])
        h = runs["ppo-" + name + "-s0"]["hist"]
        print("    largest ratio in any minibatch", max(x["ratioMax"] for x in h), "| mean approx KL per iteration", round(sum(x["approxKL"] for x in h) / len(h), 5))
        gc.collect(); torch.cuda.empty_cache()
