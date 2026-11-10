"""Check the judge on your own model's replies: label a small sample by reading it, then measure agreement.

  python -m capstone.handlabel export --n 40      writes <work>/handlabel/sheet.jsonl (replies shown as A and B in a
                                                  random order, the model names hidden in a separate key file)
  (fill "label" with A, B or C in each line of sheet.jsonl, by reading the request and both replies)
  python -m capstone.handlabel score              agreement of each judge with your labels, with a Wilson interval

HelpSteer3 checks the judge on other models' replies. This sheet checks it on the replies it actually scores.
Pick prompts at random, not the ones that look interesting, and label before you look at the judge's verdicts.
"""
import random
import sys

from .common import cli, done, path, read_json, read_jsonl, write_json, write_jsonl
from .stats import wilson


def main():
    mode = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("export", "score") else "score"
    a = cli("hand labels", [(("--n",), dict(type=int, default=40)), (("--x",), dict(default="dpo")), (("--y",), dict(default="official")),
                            (("--max-tokens",), dict(type=int, default=600))])
    sheet, key = path(a, "handlabel", "sheet.jsonl"), path(a, "handlabel", "key.json")
    arena = read_jsonl(path(a, "data", "arena.jsonl"))
    gx = [r[0] for r in read_json(path(a, "eval", "gen_%s_arena.json" % a.x))["results"]]
    gy = [r[0] for r in read_json(path(a, "eval", "gen_%s_arena.json" % a.y))["results"]]
    if mode == "export":
        if done(sheet):
            print("sheet exists; delete it to draw a new sample")
            return
        rng = random.Random(a.seed + 7)
        # Short replies only, so a reader can judge both in full. This biases the sample, and the report says so.
        idx = [i for i in range(len(arena)) if gx[i]["tokens"] <= a.max_tokens and gy[i]["tokens"] <= a.max_tokens
               and gx[i]["finished"] and gy[i]["finished"]]
        pick = rng.sample(idx, min(a.n, len(idx)))
        rows, keys = [], {}
        for i in pick:
            flip = rng.random() < 0.5
            A, B = (gy[i], gx[i]) if flip else (gx[i], gy[i])
            rows.append({"uid": arena[i]["uid"], "category": arena[i]["category"], "prompt": arena[i]["prompt"],
                         "A": A["text"], "B": B["text"], "label": ""})
            keys[arena[i]["uid"]] = {"A": a.y if flip else a.x, "B": a.x if flip else a.y}
        write_jsonl(sheet, rows)
        write_json(key, {"eligible": len(idx), "n": len(rows), "maxTokens": a.max_tokens, "keys": keys})
        print("wrote", len(rows), "of", len(idx), "eligible prompts to", sheet)
        return
    labels = {r["uid"]: r["label"].strip().upper() for r in read_jsonl(sheet) if r["label"].strip()}
    keys = read_json(key)["keys"]
    out = {"n": len(labels), "humanWinsForX": None, "judges": {}}
    xwin = [1.0 if keys[u].get(l) == a.x else 0.5 if l == "C" else 0.0 for u, l in labels.items()]
    out["humanWinsForX"] = round(sum(xwin) / max(1, len(xwin)), 4)
    import glob
    import os
    for f in glob.glob(path(a, "judge", "compare_%s_vs_%s_*.json" % (a.x, a.y))):
        c = read_json(f)
        per = {p["uid"]: p["score"] for p in c["perPrompt"]}
        agree, n = 0, 0
        for u, l in labels.items():
            human = 1.0 if keys[u].get(l) == a.x else 0.5 if l == "C" else 0.0
            judge = per[u]                      # 1, 0.75, 0.5, 0.25 or 0 for X
            n += 1
            agree += (human > 0.5 and judge > 0.5) or (human < 0.5 and judge < 0.5) or (human == 0.5 and judge == 0.5)
        out["judges"][os.path.basename(f)] = {"agree": agree, "n": n, "share": round(agree / max(1, n), 4), "ci": wilson(agree, n)}
    write_json(path(a, "handlabel", "agreement.json"), out)
    print(out)


if __name__ == "__main__":
    main()
