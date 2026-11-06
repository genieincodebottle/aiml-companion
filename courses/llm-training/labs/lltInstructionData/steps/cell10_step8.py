def run_arm(name, seed=0):
    # Train one arm, score it on IFEval, keep the per-prompt scores, free the GPU.
    model, tok = train_arm([pool[i] for i in arms[name]], name, seed=seed)
    loose, strict, stopped, texts = ifeval_scores(model, tok)
    results[name] = loose
    print(name.ljust(11), "| IFEval prompt-level loose", round(loose.mean(), 3), "| strict", round(strict.mean(), 3),
          "| answers that ended with the end token", round(stopped, 3))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results = {}
run_arm("curated-1k")
