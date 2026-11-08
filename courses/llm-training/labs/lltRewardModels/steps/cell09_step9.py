import random

def rb2_subset(per_subset=60, seed=0):
    # 60 items from each of the five subsets that score one chosen reply against three rejected ones
    ds = load_dataset("allenai/reward-bench-2", split="test")
    rng, items = random.Random(seed), []
    for sub in ["Factuality", "Precise IF", "Math", "Safety", "Focus"]:
        idx = [i for i, s in enumerate(ds["subset"]) if s == sub]
        items += [dict(ds[i]) for i in sorted(rng.sample(idx, per_subset))]
    return items

items = rb2_subset()
seqs = [chat_ids(it["prompt"], reply)[:2048] for it in items for reply in it["chosen"][:1] + it["rejected"][:3]]
s = np.array(score(rm, seqs)).reshape(len(items), 4)
correct = s[:, 0] > s[:, 1:].max(1)
for sub in ["Factuality", "Precise IF", "Math", "Safety", "Focus"]:
    print(sub, round(float(np.mean([c for c, it in zip(correct, items) if it["subset"] == sub])), 3))
print("mean of the five", round(float(correct.mean()), 3), "| random guessing 0.25")
