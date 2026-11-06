if "curated-1k" not in globals().get("results", {}):
    raise RuntimeError("Run cell 12 first, or skip this cell and run cell 14, which compares our recorded scores.")

def paired_bootstrap(a, b, n_boot=10000, seed=0):
    # Mean of b - a over the same IFEval prompts, with a 95% interval from resampling prompts.
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

run_arm("bulk-1k-s0")
diff, lo, hi = paired_bootstrap(results["bulk-1k-s0"], results["curated-1k"])
print("curated minus bulk, IFEval loose:", round(diff, 3), "| 95% interval", round(lo, 3), "to", round(hi, 3))
