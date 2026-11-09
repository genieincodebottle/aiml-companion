"""Build llt6/lltDPO.tryit_data.json from the recorded core kernel (per-pair held-out log-prob changes per beta run).
  python tryit_data.py <llt_dpo_core.json> <out.json>
"""
import json, sys
rec = json.load(open(sys.argv[1], encoding="utf-8"))
runs = rec["runs"]
arms = {"0.1": {"chosen": [round(x, 2) for x in rec["finalHeld"]["dc"]], "rejected": [round(x, 2) for x in rec["finalHeld"]["dr"]],
                "runId": "lltDPO-T2-dpo-b0.1-s0", "held": runs["dpo-b0.1-s0"]["held"]}}
for b, k in [("0.01", "dpo-b0.01-s0"), ("0.5", "dpo-b0.5-s0")]:
    arms[b] = {"chosen": runs[k]["perPair"]["chosen"], "rejected": runs[k]["perPair"]["rejected"], "runId": "lltDPO-T2-" + k, "held": runs[k]["held"]}
out = {"source": "runs/lltDPO (kernel llt-dpo-core v1, 2026-11-08); 300 held-out UltraFeedback test pairs; policy minus SFT reference "
                  "log-prob of each reply in nats after 125 DPO steps", "trainedAtBeta": arms,
       "goldWinVsSFT": {"0.01": runs["dpo-b0.01-s0"]["vsSFT"], "0.1": runs["judge"], "0.5": runs["dpo-b0.5-s0"]["vsSFT"]}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], {b: len(a["chosen"]) for b, a in arms.items()})
