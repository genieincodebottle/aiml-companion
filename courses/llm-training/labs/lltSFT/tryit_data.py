"""Write the Try it widget data for lltSFT from Run Library files (no new measurement).

Reads runs/lltSFT/lltSFT-t1-packing.json (the three packed rewrite conversations: ids, labels, per-variant losses)
and decodes each id with the SmolLM2 tokenizer. Writes <out>.json: per document, the tokens as
[text, id, carriesLoss, isSpecial] and the measured losses per variant.
Usage: python tryit_data.py <out.json>
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from runlog import read_run  # noqa: E402

run = read_run("lltSFT-t1-packing")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
special = set(tok.all_special_ids)
docs = []
for s in run["samples"]:
    toks = []
    for i, l in zip(s["ids"], s["labels"]):
        text = tok.decode([i])
        toks.append([text, i, l != -100, i in special])
    n_loss = sum(t[2] for t in toks)
    docs.append({"doc": s["doc"], "length": len(toks), "lossTokens": n_loss, "tokens": toks})
v = run["evals"]["variants"]
out = {
    "source": "runs/lltSFT/lltSFT-t1-packing.json (SmolLM2-135M, float32, CPU; three training conversations of "
              "smoltalk_smollm3_smol_rewrite_no_think, rows 0 to 2 of the training selection)",
    "lens": run["evals"]["lens"], "total": run["evals"]["total"],
    "positionIdsReset": run["evals"]["packedPositionIds"],
    "docs": docs,
    "lossPerDoc": {name: [r["lossPacked"] for r in rows] for name, rows in v.items()},
    "lossAlone": [r["lossAlone"] for r in v["positionReset"]],
    "maxAbsLogitDiff": {name: [r["maxAbsLogitDiff"] for r in rows] for name, rows in v.items()},
}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
print({k: out[k] for k in ("lens", "total", "lossPerDoc", "lossAlone", "maxAbsLogitDiff")},
      [(d["doc"], d["length"], d["lossTokens"]) for d in docs])
