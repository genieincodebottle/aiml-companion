"""Turn the Kaggle output of llt-rm-main into Run Library files, the best-of-n analysis,
the Hub upload folder and the Try it data. No new measurement.

  python record_gpu.py
Reads  kaggle/out/llt-rm-main/
Writes runs/lltRewardModels/*.json and the Hub upload file bon_pool.json (folder RM_HUB_MIRROR)
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

KOUT = HERE.parent.parent / "kaggle" / "out"
POOLD = TRAIND = KOUT / "llt-rm-main"
T4 = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6}


def load(p):
    return json.load(open(p, encoding="utf-8"))


def best_of_n(select, value, n):
    P, N = select.shape
    w = np.array([math.comb(i, n - 1) for i in range(N)], dtype=float) / math.comb(N, n)
    return float(np.mean([(w * value[p][np.argsort(select[p])]).sum() for p in range(P)]))


def bon_per_prompt(select, value, n):
    P, N = select.shape
    w = np.array([math.comb(i, n - 1) for i in range(N)], dtype=float) / math.comb(N, n)
    return np.array([(w * value[p][np.argsort(select[p])]).sum() for p in range(P)])


def boot_ci(x, seed=0, B=10000):
    x = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    pool_all = load(POOLD / "llt_rm_main.json")
    env = dict(pool_all["env"])
    runs = {r["tag"]: r for r in pool_all["runs"]}
    pool = load(POOLD / "pool.json")
    gold = np.array(load(POOLD / "gold_pool.json")["scores"])
    P, N = len(pool["prompts"]), pool["n"]
    G = gold.reshape(P, N)
    toks = np.array([s["newTokens"] for s in pool["samples"]]).reshape(P, N)
    ended = np.array([s["ended"] for s in pool["samples"]]).reshape(P, N)
    s, g = runs["pool-sample"], runs["pool-gold"]
    write_run("lltRewardModels-T2-pool", "T2", "baseline",
              {"policy": pool["policy"], "dtype": "float32", "prompts": P, "n": N, "newTokens": pool["newTokens"],
               "temperature": 1.0, "top_p": 1.0, "top_k": 0, "promptRule": "first 64 test_prefs prompts, prompt <= 200 tokens, chosen reference <= 160 tokens",
               "gold": "Skywork/Skywork-Reward-V2-Qwen3-1.7B fp16, model-card chat format"},
              0, [], evals={"sample": {k: s.get(k) for k in ("completions", "totalGenSec", "batchSec", "peakMemGB", "endedShare", "meanNewTokens", "loadSec")},
                            "gold": {k: g.get(k) for k in ("loadSec", "residentGB", "peakMemGB", "poolSec", "poolSeqPerSec", "heldSec", "heldOut",
                                                           "heldOutCapped", "rb2", "rb2Mean", "rb2Sec", "exampleRendered", "padToken")},
                            "goldPoolMean": float(gold.mean()), "goldPoolStd": float(gold.std()),
                            "goldVsLengthPearson": float(np.corrcoef(gold, toks.ravel())[0, 1]),
                            "goldEndedMean": float(G[ended].mean()), "goldCappedMean": float(G[~ended].mean())},
              wall_clock_sec=s.get("wallSec", 0) + g.get("wallSec", 0), hardware=T4, env=env,
              notes="Kaggle kernel llt-rm-main v1. Gold held-out accuracy uses the gold model's own chat format.")

    tr_all = pool_all
    tenv = dict(tr_all["env"])
    tr = {r["tag"]: r for r in tr_all["runs"]}
    c = tr["trl-check"]
    write_run("lltRewardModels-T2-trl-check", "T2", "baseline",
              {"pairs": 64, "batchPairs": 4, "steps": 40, "lr": 1e-5, "max_length": 1024, "fp16": True},
              0, [{"step": i + 1, "trainLoss": h.get("loss"), "accuracy": h.get("accuracy"), "margin": h.get("margin")} for i, h in enumerate(c.get("log", []))],
              evals={k: c.get(k) for k in ("lossCheck", "idsEqual", "trainableDtypes", "trlKeptPairs", "pairsGiven", "trainSec", "peakMemGB", "trl")},
              wall_clock_sec=c.get("wallSec"), hardware=T4, env=tenv, notes="Kaggle kernel llt-rm-main v1")

    bon = {"n": [1, 2, 4, 8, 16, 32, 64], "kl": [math.log(n) - (n - 1) / n for n in [1, 2, 4, 8, 16, 32, 64]], "proxies": {}}
    nfull = list(range(1, N + 1))
    bon["goldByGold"] = [best_of_n(G, G, n) for n in nfull]
    bon["tokensByGold"] = [best_of_n(G, toks, n) for n in nfull]
    for tag in ("rm-s0", "rm-s1", "rm-bal-s0", "rm-lr1e-4"):
        r = tr.get(tag)
        if not r:
            continue
        evals = r.get("evals", [])
        write_run("lltRewardModels-T2-" + tag, "T2", "baseline" if tag in ("rm-s0", "rm-s1") else
                  ("ablation:length-balanced" if tag == "rm-bal-s0" else "broken:lr-1e-4"),
                  {k: r["job"].get(k) for k in ("pairs", "heldPairs", "lr", "batchPairs", "balanced", "evalFractions")} |
                  {"model": "HuggingFaceTB/SmolLM2-360M + score head", "maxLen": 1024, "optimizer": "AdamW wd 0", "schedule": "3% warmup, cosine",
                   "precision": "fp32 weights, fp16 autocast, GradScaler 2048", "gradientCheckpointing": True},
                  r["job"]["seed"], r.get("metrics", []),
                  evals={"data": r.get("data"), "evals": evals, "rb2": r.get("rb2"), "rb2Mean": r.get("rb2Mean"), "head": r.get("head"),
                         "trainableParams": r.get("trainableParams"), "trainSec": r.get("trainSec"), "sPerStepMedian": r.get("sPerStepMedian"),
                         "peakMemGB": r.get("peakMemGB"), "nonFiniteGradSteps": r.get("nonFiniteGradSteps"), "oomRetries": r.get("oomRetries")},
                  wall_clock_sec=r.get("wallSec"), hardware=T4, env=tenv, notes="Kaggle kernel llt-rm-main v1")
        for e in evals:
            f = TRAIND / ("proxy_pool_%s_step%d.json" % (tag, e["step"]))
            if not f.exists():
                continue
            Pr = np.array(load(f)["scores"]).reshape(P, N)
            key = "%s@%d" % (tag, e["step"])
            curve = {"pairsSeen": e["pairsSeen"], "heldOutAcc": e["heldOut"]["acc"],
                     "proxy": [best_of_n(Pr, Pr, n) for n in nfull], "gold": [best_of_n(Pr, G, n) for n in nfull],
                     "tokens": [best_of_n(Pr, toks, n) for n in nfull],
                     "proxyGoldSpearmanWithinPrompt": float(np.mean([np.corrcoef(np.argsort(np.argsort(Pr[p])), np.argsort(np.argsort(G[p])))[0, 1] for p in range(P)])),
                     "proxyLengthPearson": float(np.corrcoef(Pr.ravel(), toks.ravel())[0, 1])}
            d = bon_per_prompt(Pr, G, 64) - bon_per_prompt(Pr, G, 1)
            curve["gold64minus1"] = {"mean": float(d.mean()), "ci": boot_ci(d)}
            g1, g4 = bon_per_prompt(Pr, G, 1), bon_per_prompt(Pr, G, 4)
            d4 = g4 - g1
            curve["gold4minus1"] = {"mean": float(d4.mean()), "ci": boot_ci(d4)}
            d644 = bon_per_prompt(Pr, G, 64) - g4
            curve["gold64minus4"] = {"mean": float(d644.mean()), "ci": boot_ci(d644)}
            curve["peakNote"] = "goldPeak picks the peak on the same data it tests, so peakMinus64 is biased upward"
            k = int(np.argmax(curve["gold"]))
            dpk = bon_per_prompt(Pr, G, k + 1) - bon_per_prompt(Pr, G, 64)
            curve["goldPeak"] = {"n": k + 1, "gold": curve["gold"][k], "peakMinus64": {"mean": float(dpk.mean()), "ci": boot_ci(dpk)}}
            # Gao et al. 2023 best-of-n form: gold(d) - gold(0) = d (alpha - beta d), d = sqrt(KL)
            dd = np.sqrt([math.log(n) - (n - 1) / n for n in nfull])
            y = np.array(curve["gold"]) - curve["gold"][0]
            A = np.stack([dd, -dd ** 2], 1)
            (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
            curve["gaoFit"] = {"alpha": float(alpha), "beta": float(beta), "peakD": float(alpha / (2 * beta)) if beta > 0 else None}
            bon["proxies"][key] = curve
    write_run("lltRewardModels-T2-bon", "T2", "baseline",
              {"estimator": "unbiased best-of-n over N=64 recorded samples per prompt", "prompts": P, "N": N,
               "kl": "log n - (n - 1) / n nats"}, 0, [], evals=bon, hardware=T4, env=tenv,
              notes="Computed locally from the llt-rm-main outputs; no new GPU time")

    # Hub file for step 8: prompts, samples and the gold score of each sample
    hub = Path(os.environ.get("RM_HUB_MIRROR", "hub_mirror")) / "lltRewardModels"   # upload folder, outside the repo
    hub.mkdir(parents=True, exist_ok=True)
    json.dump({"policy": pool["policy"], "n": N, "newTokens": pool["newTokens"], "temperature": 1.0,
               "gold_model": "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "prompts": pool["prompts"], "samples": pool["samples"],
               "gold": [float(x) for x in gold]}, open(hub / "bon_pool.json", "w", encoding="utf-8"))
    print("wrote", hub / "bon_pool.json")
    print(json.dumps({k: {kk: (vv if not isinstance(vv, list) else [round(x, 3) for x in vv[:1] + [vv[i - 1] for i in (2, 4, 8, 16, 32, 64)]])
                          for kk, vv in v.items()} for k, v in bon["proxies"].items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
