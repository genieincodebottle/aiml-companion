"""lltPretrainData exact measurements on CPU (small inputs, NumPy only). Each section writes one Run
Library JSON to runs/lltPretrainData/.

  lltPretrainData-t0-gopher-fixtures   the Gopher and C4 rules on hand-made fixtures, pass and fail
  lltPretrainData-t0-lsh-scurve        MinHash LSH candidate rate vs 1 - (1 - J**r)**b, simulated
  lltPretrainData-t0-decontam-paraphrase  13-gram and 8-gram decontamination on verbatim vs paraphrased ARC items
  lltPretrainData-t0-mixture-accounting   weighting by documents vs tokens, and effective epochs per source

Usage: python measure.py <arc_easy_test.parquet> [prep_stats.json]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptd  # noqa: E402
from paraphrases import ITEMS  # noqa: E402
from runlog import write_run  # noqa: E402


def fixtures():
    good = ("The library opens at nine every morning and closes at six in the evening. " * 2 +
            "Children come after school to read stories, and older readers sit near the back with the newspapers. "
            "On rainy days the reading room is full, and the staff put out extra chairs. "
            "The building was restored last year with a new roof and better lighting for the tables. "
            "Volunteers run a book club on the first Monday of each month, and anyone can join it.")
    cases = {
        "good_page": (good, None),
        "too_short": ("The cat sat on the mat and looked at the door.", "too_few_words"),
        "hash_tags": (" ".join(["#deal #sale #shop the best and to of"] * 12), "hash_ratio"),
        "no_stop_words": (" ".join(["buy cheap phone case red blue green black white grey pink"] * 6), "stop_words"),
        "symbols": (" ".join(["1234 5678 9012 3456 7890 the of and cat dog"] * 7), "alpha_words"),
        "bullets": ("\n".join(["- item %d of the list and the rest" % i for i in range(60)]), "bullet_lines"),
        "repeated_lines": ("\n".join(["Click here to subscribe to the newsletter and the offers."] * 20 + [good]), "dup_lines"),
    }
    out = {}
    for name, (text, want) in cases.items():
        got = ptd.gopher_quality(text) or ptd.gopher_repetition(text)
        out[name] = {"expected": want, "got": got, "ok": got == want}
    c4_cases = {
        "c4_keeps_good": (good, None),
        "c4_lorem": (good + "\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do.", "lorem_ipsum"),
        "c4_curly": (good + "\nfunction f() { return the value of the call; }.", "curly_bracket"),
        "c4_two_sentences": ("This page has one sentence here. It has a second sentence too.", "fewer_than_3_sentences"),
    }
    for name, (text, want) in c4_cases.items():
        _, got = ptd.c4_clean(text)
        out[name] = {"expected": want, "got": got, "ok": got == want}
    cleaned, _ = ptd.c4_clean(good + "\nShare this page\nWe use cookies to improve your experience on this website today.\nRead more")
    out["c4_drops_lines"] = {"keptLines": cleaned.count("\n") + 1, "ok": "cookies" not in cleaned and "Share this" not in cleaned}
    scrubbed, ne, ni = ptd.scrub_pii("Write to jane.doe@example.org or ping 8.8.8.8, not 192.168.1.1.")
    out["pii"] = {"text": scrubbed, "emails": ne, "ips": ni, "ok": ne == 1 and ni == 1 and "192.168.1.1" in scrubbed}
    for k, v in out.items():
        print(k, v)
    assert all(v["ok"] for v in out.values()), "a fixture failed"
    write_run("lltPretrainData-t0-gopher-fixtures", "T0", "check", {"rules": "ptd.py"}, None, [], evals=out)


def make_pair(J, size, rng):
    """Two sets of random 31-bit shingle values (what blake2b gives real shingles) with
    |A & B| / |A | B| equal to J, rounded to the nearest whole overlap."""
    inter = int(round(2 * size * J / (1 + J)))
    vals = rng.choice(2**31 - 1, size=2 * size - inter, replace=False).astype(np.uint64)
    a = vals[:size]
    b = np.concatenate([vals[:inter], vals[size:]])
    return a, b, inter / (2 * size - inter)


def lsh_scurve():
    rng = np.random.default_rng(0)
    res = []
    for bands, rows in [(14, 8), (20, 5)]:
        a, b = ptd.make_hashes(bands * rows, seed=1)
        for J in [0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
            hits, est = 0, []
            trials = 400
            for t in range(trials):
                s1, s2, jx = make_pair(J, 200, rng)
                m1, m2 = ptd.minhash(s1, a, b), ptd.minhash(s2, a, b)
                eq = m1 == m2
                est.append(eq.mean())
                hits += any(eq[k * rows:(k + 1) * rows].all() for k in range(bands))
            p = ptd.p_candidate(jx, bands, rows)
            emp = hits / trials
            se = (p * (1 - p) / trials) ** 0.5
            row = {"bands": bands, "rows": rows, "J": round(jx, 4), "formula": round(p, 4), "simulated": round(emp, 4),
                   "se": round(se, 4), "within3se": abs(emp - p) <= 3 * se + 1e-9,
                   "meanEstimate": round(float(np.mean(est)), 4), "estimateStd": round(float(np.std(est)), 4),
                   "theoryStd": round((jx * (1 - jx) / (bands * rows)) ** 0.5, 4), "trials": trials}
            print(row); res.append(row)
    thr = {"14x8": round((1 / 14) ** (1 / 8), 4), "20x5": round((1 / 20) ** (1 / 5), 4)}
    print("threshold (1/b)**(1/r)", thr)
    assert sum(r["within3se"] for r in res) >= len(res) - 1, "simulation disagrees with the formula"
    write_run("lltPretrainData-t0-lsh-scurve", "T0", "check", {"setSize": 200, "trials": 400, "hash": "(a*x+b) mod 2**31-1"},
              0, res, evals={"thresholdApprox": thr})


def lsh_structured():
    """The same check with STRUCTURED shingle values (consecutive integers) fed straight into the
    affine hashes. Affine hashes are not min-wise independent on such inputs, so the estimates and
    the candidate rate fall below the formula. Real shingles go through blake2b first."""
    rng = np.random.default_rng(0)
    a, b = ptd.make_hashes(112, seed=1)
    res = []
    for J in [0.7, 0.75, 0.8]:
        hits, est = 0, []
        for t in range(400):
            size = 200
            inter = int(round(2 * size * J / (1 + J)))
            start = t * 10**7
            s1 = np.arange(start, start + size).astype(np.uint64)
            s2 = np.concatenate([s1[:inter], np.arange(start + 10**6, start + 10**6 + size - inter).astype(np.uint64)])
            m1, m2 = ptd.minhash(s1, a, b), ptd.minhash(s2, a, b)
            eq = m1 == m2
            est.append(eq.mean()); hits += any(eq[k * 8:(k + 1) * 8].all() for k in range(14))
        jx = inter / (2 * size - inter)
        row = {"J": round(jx, 4), "formula": round(ptd.p_candidate(jx, 14, 8), 4), "simulated": hits / 400,
               "meanEstimate": round(float(np.mean(est)), 4)}
        print("structured", row); res.append(row)
    write_run("lltPretrainData-t0-lsh-structured-inputs", "T0", "broken:minhash-on-unhashed-shingles",
              {"setSize": 200, "trials": 400, "bands": 14, "rows": 8, "shingles": "consecutive integers, no blake2b"}, 0, res)


def decontam(arc_path):
    import pandas as pd
    arc = pd.read_parquet(arc_path)
    texts = [q + " " + " ".join(c["text"]) for q, c in zip(arc["question"], arc["choices"])]
    out = {}
    filler = "Posted in Science Revision. Here is today's practice question for the class. "
    for n in (13, 8):
        idx = ptd.build_eval_index(texts, n)
        verb = sum(ptd.contaminated(filler + v, idx, n) for _, v, _ in ITEMS)
        para = sum(ptd.contaminated(filler + p, idx, n) for _, _, p in ITEMS)
        # False positives on clean text: the same filler plus unrelated sentences
        clean = ["The library opens at nine every morning and closes at six in the evening, and the staff put out chairs.",
                 "Our recipe uses two cups of flour, one egg and a pinch of salt, and it bakes in twenty minutes."]
        fp = sum(ptd.contaminated(filler + c, idx, n) for c in clean)
        out[str(n)] = {"items": len(ITEMS), "verbatimCaught": verb, "paraphraseCaught": para, "cleanFlagged": fp, "cleanPages": len(clean),
                       "indexNgrams": len(idx)}
        print(n, out[str(n)])
    write_run("lltPretrainData-t0-decontam-paraphrase", "T0", "broken:decontam-misses-paraphrase",
              {"evalSet": "ARC-Easy test, question + choices", "items": len(ITEMS)}, None, [], evals=out)


def mixture_accounting(stats_path):
    s = json.load(open(stats_path))
    corp = s["corpora"]
    mean = {"web": corp["F_train"]["meanDocTokens"], "wiki": corp["wiki_train"]["meanDocTokens"],
            "stories": corp["ts_train"]["meanDocTokens"]}
    w = {"web": 1 / 3, "wiki": 1 / 3, "stories": 1 / 3}
    tot = sum(w[k] * mean[k] for k in w)
    by_docs = {k: round(w[k] * mean[k] / tot, 4) for k in w}
    print("mean doc tokens", mean, "| token share when weighting by documents", by_docs)
    budget = 9_830_400  # 300 steps x 64 x 512, the mixture runs
    unique = {"web": corp["F_train"]["tokens"], "wiki": corp["wiki_train"]["tokens"], "stories": corp["ts_train"]["tokens"]}
    eff = {k: round(w[k] * budget / unique[k], 4) for k in w}
    write_run("lltPretrainData-t0-mixture-accounting", "T0", "broken:weights-by-documents",
              {"weights": w, "budgetTokens": budget}, None, [],
              evals={"meanDocTokens": mean, "tokenShareByDocuments": by_docs, "uniqueTokens": unique, "effectiveEpochs": eff})


if __name__ == "__main__":
    fixtures()
    lsh_scurve()
    lsh_structured()
    decontam(sys.argv[1])
    if len(sys.argv) > 2:
        mixture_accounting(sys.argv[2])
