"""Recompute the train/validation leak of the unfiltered corpus against the pages the unfiltered
model actually trained on. prep_ptd.py measured the leak against the whole training side of the
shard (142,410 pages); the model trained on the first 24,346 of them (60M tokens). This script
rebuilds the same shuffle and split, repeats the full measurement as a check, then measures the
leak against the 24,346-page slice, overall and for the 88 pages saved as U_val_leak.

Writes run lltPretrainData-cpu-leak-check. Usage: python leak_check.py <c4noclean_00000.json.gz>
"""
import gzip, json, os, random, sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptd  # noqa: E402
from runlog import write_run  # noqa: E402

A, B = ptd.make_hashes(112, seed=1)


def sig(t):
    return ptd.minhash(ptd.shingles(t), A, B)


def leaked(hold_sigs, train_sigs, hold, train):
    allsig = np.concatenate([hold_sigs, train_sigs])
    n = len(hold_sigs)
    near = set()
    for a, b in ptd.lsh_candidates(allsig, 14, 8):
        if a < n <= b and float((allsig[a] == allsig[b]).mean()) >= 0.8:
            near.add(a)
    keys = {ptd.exact_key(x) for x in train}
    exact = {i for i, x in enumerate(hold) if ptd.exact_key(x) in keys}
    return near, exact


if __name__ == "__main__":
    raw = []
    with gzip.open(sys.argv[1], "rt", encoding="utf-8") as f:
        for line in f:
            raw.append(json.loads(line)["text"])
    random.Random(0).shuffle(raw)
    hold, utrain = raw[:6000], raw[6000:]
    N_SLICE = 24346  # cpu-corpora U_train docs: utrain[:40000] cut at 60M tokens
    with Pool(4) as pool:
        hs = np.stack(pool.map(sig, hold, chunksize=64))
        ts = np.stack(pool.map(sig, utrain, chunksize=64))
    near_all, exact_all = leaked(hs, ts, hold, utrain)
    leak_all = sorted(near_all | exact_all)
    near_s, exact_s = leaked(hs, ts[:N_SLICE], hold, utrain[:N_SLICE])
    leak_s = set(near_s | exact_s)
    saved = leak_all[:88]  # U_val_leak holds the first 88 leaked pages (600k-token cap)
    res = {"heldOut": 6000, "trainSideFull": len(utrain), "trainSlice": N_SLICE,
           "leakFull": len(leak_all), "nearFull": len(near_all), "exactFull": len(exact_all),
           "leakVsSlice": len(leak_s), "nearVsSlice": len(near_s), "exactVsSlice": len(exact_s),
           "savedLeakPages": len(saved), "savedLeakPagesWithDupInSlice": sum(i in leak_s for i in saved)}
    print(json.dumps(res))
    write_run("lltPretrainData-cpu-leak-check", "T1", "check",
              {"source": "C4 en.noclean shard 00000, shuffle random.Random(0), 6,000 held out", "minhash": "5-gram, 14 bands x 8 rows, est J >= 0.8"},
              0, [], evals=res)
