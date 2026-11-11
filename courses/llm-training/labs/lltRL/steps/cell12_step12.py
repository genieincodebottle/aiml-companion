RUN_EXPERIMENT = False   # True trains six more 150-step policies, about 8 h on a T4 (6 x 80 min, projected). False: read our runs below
ARMS = {"G4":      dict(GRPO, prompts=8, G=4),            # same 32 replies per step, smaller groups
        "G16":     dict(GRPO, prompts=2, G=16),           # bigger groups, fewer prompts
        "beta0.04": dict(GRPO, beta=0.04),                # KL penalty to the base model, the DeepSeekMath setting
        "seqmean": dict(GRPO, loss_type="grpo"),          # the original GRPO average: per reply, then over replies
        "drgrpo":  dict(GRPO, loss_type="dr_grpo", adv="drgrpo"),   # no std scaling, constant normalizer
        "rloo":    dict(GRPO, adv="rloo")}                # leave-one-out baseline instead of the group mean

if RUN_EXPERIMENT:
    del grpo_model; gc.collect(); torch.cuda.empty_cache()
    for name, cfg in ARMS.items():
        m, runs["grpo-" + name] = train_grpo(cfg, tag="grpo-" + name)
        runs["grpo-" + name]["final"] = evaluate_gsm(m)
        h = runs["grpo-" + name]["hist"]
        last, first = h[-50:], h[:50]
        print(name, "| reward, last 50 steps", round(sum(x["reward"] for x in last) / len(last), 3), "| reply tokens, first and last 50",
              round(sum(x["meanLen"] for x in first) / len(first), 1), round(sum(x["meanLen"] for x in last) / len(last), 1),
              "| zero-spread groups", round(sum(x["zeroStd"] for x in h) / len(h), 3), "| greedy accuracy", runs["grpo-" + name]["final"]["greedy"])
        del m; gc.collect(); torch.cuda.empty_cache()
