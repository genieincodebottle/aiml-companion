RUN_EXPERIMENT = False   # True trains three more DPO models on your T4; our runs are in the table of this step

def experiment(arms, pairs, ref):
    # arms: (name, loss, beta, seed, extra). Train each arm, sample from it, free it; then score every arm against SFT.
    for name, kind, beta, seed, extra in arms:
        model, info, last = train_pref(pairs, ref, kind, beta=beta, seed=seed, **extra)
        gens[name] = generate(model, eval_prompts)
        info["final"] = {k: v for k, v in last.items() if k not in ("dc", "dr")}
        info["perPair"] = {"chosen": np.round(last["dc"], 2).tolist(), "rejected": np.round(last["dr"], 2).tolist()}
        runs[name] = info
        del model
        gc.collect()
        torch.cuda.empty_cache()
    sets = {a[0]: list(zip(eval_prompts, [g["text"] for g in gens[a[0]]])) for a in arms}
    scores, scores2 = gold_scores(sets), gold_scores(sets, JUDGE2)
    for name, kind, beta, seed, extra in arms:
        r, gold[name], judge2[name] = runs[name], scores[name], scores2[name]
        r["gold"], r["vsSFT"] = round(float(scores[name].mean()), 3), paired(scores[name], gold["sft"])
        r["vsSFT2"] = paired(scores2[name], judge2["sft"])
        r["tokens"] = round(float(np.mean([g["tokens"] for g in gens[name]])), 1)
        r["ended"] = round(float(np.mean([g["ended"] for g in gens[name]])), 3)
        print(name, "| held-out chosen", r["final"]["chosen"], "rejected", r["final"]["rejected"], "accuracy", r["final"]["accuracy"],
              "| win vs SFT", r["vsSFT"]["win"], r["vsSFT"]["winCI"], "| second judge", r["vsSFT2"]["win"], r["vsSFT2"]["winCI"], "| reply tokens", r["tokens"], "| ended", r["ended"])

if RUN_EXPERIMENT:
    experiment([("dpo-b0.01-s0", "dpo", 0.01, 0, {}), ("dpo-b0.5-s0", "dpo", 0.5, 0, {}), ("dpo-b0.1-s1", "dpo", 0.1, 1, {})],
               dpo_pairs, ref_dpo)
