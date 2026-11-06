"""Build the small JSON the lltLoRA overview (and its Try it) needs, from our Run Library files.

Usage: python overview_data.py <full run id with -svd file> <truncation run id> <out.json> [lora run ids...]
Everything in the output is copied or derived from runs/lltLoRA/*.json; nothing is typed by hand.
"""
import json, os, statistics as st, sys

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "lltLoRA")


def load(run_id):
    return json.load(open(os.path.join(RUNS, run_id + ".json"), encoding="utf-8"))


def main():
    full_id, trunc_id, out = sys.argv[1:4]
    lora_ids = sys.argv[4:]
    svd = load(full_id + "-svd")
    lin = svd["linear"]
    ranks = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
    kinds = {}
    for n, v in lin.items():
        kinds.setdefault(n.split(".")[-1], []).append(v)
    data = {
        "source": {"svd": full_id + "-svd", "truncation": trunc_id, "lora": lora_ids},
        "matrices": len(lin),
        "medianEnergy": {r: round(st.median(v["energy"][r] for v in lin.values()), 4) for r in ranks},
        "r90": {"median": st.median(v["r90"] for v in lin.values()), "min": min(v["r90"] for v in lin.values()),
                "max": max(v["r90"] for v in lin.values())},
        "byKind": {k: {"count": len(vs), "medianEnergy16": round(st.median(v["energy"]["16"] for v in vs), 4),
                       "medianR90": st.median(v["r90"] for v in vs)} for k, vs in kinds.items()},
        "pooled": {},
    }
    for n, p in svd["pooled"].items():
        v = lin[n]
        data["pooled"][n] = {"shape": p["shape"], "fro": v["fro"], "relChange": v["relChange"], "r90": v["r90"],
                             "energy": v["energy"], "sTop64": p["s"], "sAll": v["s"],
                             "U64": p["U64"], "V64": p["V64"], "pooledFull": p["pooledFull"]}
    tr = load(trunc_id)
    data["truncation"] = [{k: e.get(k) for k in ("rank", "heldoutLoss", "wikiLoss")} for e in tr["evals"]["truncation"]]
    data["truncationFinal"] = {k: tr["evals"]["final"].get(k) for k in ("heldoutLoss", "wikiLoss", "accStrict")}
    data["lora"] = []
    for rid in lora_ids:
        r = load(rid)
        f = r["evals"]["final"] or {}
        data["lora"].append({"runId": rid, "r": r["config"].get("r"), "targets": r["config"].get("targets"),
                             "lr": r["config"].get("lr"), "seed": r["config"].get("seed"),
                             "trainable": r.get("trainableParams"), "heldoutLoss": f.get("heldoutLoss"),
                             "wikiLoss": f.get("wikiLoss"), "accStrict": f.get("accStrict"), "peakMemGB": r.get("peakMemGB")})
    json.dump(data, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
