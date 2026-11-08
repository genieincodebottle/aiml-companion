import gc
RUN_EXPERIMENT = False    # True trains two more reward models on your T4; our runs are in the table of this step

def select_balanced(n):
    # same shuffled stream as step 3, but half the pairs have the longer reply chosen and half the shorter one
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=0)
    out, count = [], {True: 0, False: 0}
    for ex in ds:
        c, r = ex["chosen"][-1]["content"], ex["rejected"][-1]["content"]
        if c == r:
            continue
        ci, ri = chat_ids(ex["prompt"], c), chat_ids(ex["prompt"], r)
        if len(ci) > MAXLEN or len(ri) > MAXLEN or len(ci) == len(ri) or count[len(ci) > len(ri)] == n // 2:
            continue
        count[len(ci) > len(ri)] += 1
        out.append({"prompt": ex["prompt"], "chosen": c, "rejected": r, "c": ci, "r": ri,
                    "sc": ex["score_chosen"], "sr": ex["score_rejected"]})
        if len(out) == n:
            return out

if RUN_EXPERIMENT:
    rm_cpu = rm.cpu()                         # keep step 5's model, but off the GPU
    gc.collect()
    torch.cuda.empty_cache()
    for name, pairs, seed in [("seed 1", train, 1), ("length-balanced, seed 0", select_balanced(3000), 0)]:
        m = train_rm(pairs, seed=seed)
        print(name)
        held_out_report(m, held)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    rm = rm_cpu.to(GPU)
