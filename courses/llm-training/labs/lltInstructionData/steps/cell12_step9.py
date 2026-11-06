import json, urllib.request
import numpy as np

# Our recorded per-prompt IFEval scores (1 = prompt passed), from the run folder linked on the Lab card.
RUNS = "https://raw.githubusercontent.com/genieincodebottle/aiml-companion/main/courses/llm-training/runs/lltInstructionData/"

def recorded(run_id):
    with urllib.request.urlopen(RUNS + run_id + ".json") as f:
        return np.array(json.load(f)["evals"]["ifevalLoosePerPrompt"], dtype=float)

def paired_bootstrap(a, b, n_boot=10000, seed=0):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

for seed in [0, 1]:
    cur = recorded("lltInstructionData-T2-sft-curated-1k-s" + str(seed))
    bulk = recorded("lltInstructionData-T2-sft-bulk-1k-s" + str(seed))
    diff, lo, hi = paired_bootstrap(bulk, cur)
    print("seed", seed, "| curated", round(cur.mean(), 3), "| bulk", round(bulk.mean(), 3),
          "| curated minus bulk", round(diff, 3), "| 95% interval", round(lo, 3), "to", round(hi, 3))
