from datasets import load_dataset
import numpy as np

MAXLEN = 1024

def select_pairs(split, n, shuffle=True):
    # Shuffle with seed 0 (training) or keep file order (held out); keep pairs whose two sequences fit MAXLEN tokens
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    if shuffle:
        ds = ds.shuffle(seed=0)
    out, read = [], 0
    for ex in ds:
        read += 1
        c, r = ex["chosen"][-1]["content"], ex["rejected"][-1]["content"]
        if c == r:
            continue
        ci, ri = chat_ids(ex["prompt"], c), chat_ids(ex["prompt"], r)
        if len(ci) <= MAXLEN and len(ri) <= MAXLEN:
            out.append({"prompt": ex["prompt"], "chosen": c, "rejected": r, "c": ci, "r": ri,
                        "sc": ex["score_chosen"], "sr": ex["score_rejected"]})
        if len(out) == n:
            return out, read

train, read_train = select_pairs("train_prefs", 3000)
held, read_held = select_pairs("test_prefs", 1000, shuffle=False)
for name, pairs, read in [("train", train, read_train), ("held-out", held, read_held)]:
    longer = np.mean([len(p["c"]) > len(p["r"]) for p in pairs])
    ties = np.mean([p["sc"] == p["sr"] for p in pairs])
    mean_len = np.mean([len(p["c"]) + len(p["r"]) for p in pairs]) / 2
    print(name, "| kept", len(pairs), "of", read, "read | chosen longer", round(float(longer), 3),
          "| equal GPT-4 scores", round(float(ties), 3), "| mean tokens", round(float(mean_len)))
print("length-only rule on held-out, pick the longer reply:", round(float(np.mean([len(p["c"]) > len(p["r"]) for p in held])), 3))
