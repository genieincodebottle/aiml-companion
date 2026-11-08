"""Write the Try it widget data (best-of-n slider) from the llt-rm-main outputs. No new measurement.

  python tryit_data.py <out.json>
Holds: the mean proxy, gold and reply-length curves for n = 1..64 (seed-0 proxy at each checkpoint, seed-1 and the
length-balanced proxy), and four example prompts with all 64 samples (text cut to 280 characters), proxy and gold scores.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "kaggle" / "out" / "llt-rm-main"
RUNS = HERE.parent.parent / "runs" / "lltRewardModels"
EXAMPLES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3]


def r3(xs):
    return [round(float(x), 3) for x in xs]


bon = json.load(open(RUNS / "lltRewardModels-T2-bon.json", encoding="utf-8"))["evals"]
pool = json.load(open(OUT / "pool.json", encoding="utf-8"))
gold = np.array(json.load(open(OUT / "gold_pool.json", encoding="utf-8"))["scores"])
P, N = len(pool["prompts"]), pool["n"]
proxy = np.array(json.load(open(OUT / "proxy_pool_rm-s0_step375.json", encoding="utf-8"))["scores"])
data = {
    "source": "runs/lltRewardModels/lltRewardModels-T2-bon (curves) and kaggle/out/llt-rm-main (samples)",
    "policy": pool["policy"], "gold": "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "proxy": "SmolLM2-360M reward model, seed 0, 3,000 pairs",
    "prompts": P, "N": N, "n": list(range(1, N + 1)),
    "klNats": [round(float(np.log(n) - (n - 1) / n), 4) for n in range(1, N + 1)],
    "goldByGold": r3(bon["goldByGold"]),
    "curves": {k: {"pairsSeen": v["pairsSeen"], "heldOutAcc": round(v["heldOutAcc"], 4), "proxy": r3(v["proxy"]),
                   "gold": r3(v["gold"]), "tokens": [round(float(x), 1) for x in v["tokens"]]}
               for k, v in bon["proxies"].items()},
    "examples": [],
}
for p in EXAMPLES:
    rows = [pool["samples"][p * N + j] for j in range(N)]
    data["examples"].append({
        "prompt": pool["prompts"][p]["prompt"][:400],
        "samples": [{"text": s["text"][:280], "tokens": s["newTokens"], "ended": s["ended"],
                     "proxy": round(float(proxy[p * N + j]), 3), "gold": round(float(gold[p * N + j]), 3)} for j, s in enumerate(rows)],
    })
Path(sys.argv[1]).write_text(json.dumps(data, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
print("wrote", sys.argv[1], Path(sys.argv[1]).stat().st_size, "bytes")
