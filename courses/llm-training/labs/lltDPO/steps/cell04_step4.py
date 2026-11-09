import math, random, time
import numpy as np

del cpu_model                          # steps 2 and 3 are done with the CPU copy
gc.collect()
MAXLEN = 512
N_SFT, N_DPO, N_HELD, N_EVAL = 3000, 1000, 300, 200

def fitting_pairs(split, shuffle):
    # UltraFeedback pairs whose chosen and rejected rows both fit MAXLEN tokens; seed-0 shuffle or file order
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    for ex in (ds.shuffle(seed=0) if shuffle else ds):
        c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
        if c == r:
            continue
        p, ci = encode(ex["prompt"], c)
        ri = encode(ex["prompt"], r)[1]
        if len(ci) <= MAXLEN and len(ri) <= MAXLEN:
            yield {"prompt": ex["prompt"], "chosen": c, "rejected": r, "p": p, "c": ci, "r": ri}

stream = fitting_pairs("train_prefs", shuffle=True)
sft_pairs = [next(stream) for _ in range(N_SFT)]          # SFT on their chosen replies
dpo_pairs = [next(stream) for _ in range(N_DPO)]          # DPO on the next 1,000, never seen in SFT
test = fitting_pairs("test_prefs", shuffle=False)
held = [next(test) for _ in range(N_HELD)]                # held-out pairs for log-prob tracking
eval_prompts = [h["prompt"] for h in held if len(h["p"]) <= 200][:N_EVAL]   # prompts for generation
for name, ps in [("SFT", sft_pairs), ("DPO", dpo_pairs), ("held-out", held)]:
    print(name, len(ps), "pairs | chosen longer", round(float(np.mean([len(x["c"]) > len(x["r"]) for x in ps])), 3),
          "| mean reply tokens, chosen", round(float(np.mean([len(x["c"]) - len(x["p"]) for x in ps]))),
          "rejected", round(float(np.mean([len(x["r"]) - len(x["p"]) for x in ps]))))
print("generation prompts", len(eval_prompts))
