"""Turn the Kaggle outputs into Run Library files (runs/lltToolUse/*.json) and the Try it data.

Usage: python make_runs.py <notebook_results.json> <tooluse_runs.json> <tryit_data.json out path> [smoke]
Needs tool_common.py (this folder) and an unpacked bfcl_eval on sys.path for re-scoring (BFCL_DIR env var).
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tool_common as C

RUNS = os.path.join(HERE, "..", "..", "runs", "lltToolUse")
nb_path, runs_path, tryit_path = sys.argv[1:4]
smoke = len(sys.argv) > 4
NB = json.load(open(nb_path, encoding="utf-8"))
RN = json.load(open(runs_path, encoding="utf-8")) if os.path.exists(runs_path) else {"runs": []}
R2 = os.path.join(os.path.dirname(runs_path), "..", "llt-tooluse-q3tok", "tooluse_runs2.json")
if os.path.exists(R2):
    RN["runs"] += json.load(open(R2, encoding="utf-8"))["runs"]
sys.path.insert(0, os.environ["BFCL_DIR"])
PLAN = {"simple_python": 4, "multiple": 4, "parallel": 4, "parallel_multiple": 4, "irrelevance": 4}
if smoke:
    PLAN = {"simple_python": 50, "multiple": 50, "parallel": 50, "parallel_multiple": 50, "irrelevance": 60}
items = C.load_bfcl(os.path.join(os.environ["BFCL_DIR"], "bfcl_eval", "data"), PLAN)
assert [it["id"] for it in items] == NB["items"], "item list differs from the recorded run"
CATS = list(PLAN)
KAGGLE = "Kaggle T4 (one GPU of T4 x2), image 2026-11-07"


def per_cat_ci(a, b):
    out = {}
    for cat in CATS:
        idx = [i for i, it in enumerate(items) if it["category"] == cat]
        out[cat] = C.paired_bootstrap([a[i] for i in idx], [b[i] for i in idx])
    out["all"] = C.paired_bootstrap(a, b)
    return out


def errors(outputs, parse):
    # why the wrong ones are wrong: malformed, no call, wrong count, wrong function, wrong arguments
    kinds = {}
    for it, text in zip(items, outputs):
        calls = parse(text, it["tools"])
        if C.check(it, calls):
            continue
        if it["category"] == "irrelevance":
            k = "called a function on an irrelevance item"
        elif calls is None:
            k = "malformed call"
        elif calls == []:
            k = "no call"
        elif len(calls) != len(it["ground_truth"]):
            k = "wrong number of calls"
        else:
            want = sorted(next(iter(e)) for e in it["ground_truth"])
            got = sorted(next(iter(c)) for c in calls)
            k = "wrong function" if want != got else "wrong or missing arguments"
        kinds.setdefault(it["category"], {}).setdefault(k, 0)
        kinds[it["category"]][k] += 1
    return kinds


def write(run_id, body):
    os.makedirs(RUNS, exist_ok=True)
    json.dump({"runId": run_id, "module": "lltToolUse", "tier": "T2", "date": "2026-11-07", **body},
              open(os.path.join(RUNS, run_id + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", run_id)


def rescore(outputs, parse):
    rows = C.score(items, outputs, parse)
    return rows, C.summary(rows)


env = NB["env"]
summ = {}
for key, parse in [("base", C.parse_xml_calls), ("sft", C.parse_xml_calls), ("baseForced", C.parse_xml_calls), ("sftForced", C.parse_xml_calls)]:
    rows, s = rescore(NB[key]["outputs"], parse)
    assert [r["correct"] for r in rows] == NB[key]["correct"], key + " re-score differs"
    summ[key] = s
own_vs_official = {k: sum(a == b for a, b in zip(NB[k]["correct"], NB[k]["officialCorrect"])) for k in ("base", "sft", "baseForced", "sftForced")}
prefix = "lltToolUse-T2-smoke-" if smoke else "lltToolUse-T2-"
common = {"hardware": {"gpu": env["gpu"], "count": 1, "vramGB": 15.6, "where": KAGGLE}, "env": env,
          "items": {"ids": NB["items"], "plan": PLAN, "source": "bfcl-eval 2026.3.23, BFCL_v4 data files"}}
nb_body = {"label": "baseline", "model": "Qwen/Qwen3.5-2B-Base", "seed": 0, **common,
           "config": {"lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target": "all-linear"}, "base": "fp16, frozen", "adapters": "fp32",
                      "lr": 2e-4, "schedule": "cosine, 10 warmup steps", "epochs": 1, "batch": "2 x 2 accumulation", "max_length": 1536,
                      "trainRows": 24 if smoke else 1200, "generation": "greedy, 384 new tokens, batch 32"},
           "check": NB["check"], "train": NB.get("train"),
           "evals": {k: {"summary": summ[k], "genSec": NB[k]["genSec"], "ownVsBfclAgree": own_vs_official[k], "errors": errors(NB[k]["outputs"], C.parse_xml_calls)}
                     for k in summ},
           "forcedSec": NB["forcedSec"],
           "pairedBootstrap": {"sftVsBase": per_cat_ci(NB["base"]["correct"], NB["sft"]["correct"]),
                               "baseForcedVsBase": per_cat_ci(NB["base"]["correct"], NB["baseForced"]["correct"]),
                               "sftForcedVsSft": per_cat_ci(NB["sft"]["correct"], NB["sftForced"]["correct"])},
           "outputs": {k: NB[k]["outputs"] for k in summ}}
write(prefix + "notebook-q35-2b-s0", nb_body)

for r in RN.get("runs", []):
    parse = C.parse_json_calls if r["model"].startswith("Qwen/Qwen3-") else C.parse_xml_calls
    rows, s = rescore(r["outputs"], parse)
    assert [x["correct"] for x in rows] == r["correct"], r["tag"] + " re-score differs"
    odd = chr(0x2B7E6)
    extra = {"repliesWithToolCallTag": sum("<tool_call>" in o for o in r["outputs"]), "repliesWithOddToken": sum(odd in o for o in r["outputs"]),
             "relevantRepliesWithJsonNameKey": sum(chr(123) + '"name"' in o for o, it in zip(r["outputs"], items) if it["category"] != "irrelevance")}
    label = {"q3-1.7b-base": "reference:base", "q3-1.7b-lora-s0": "broken:special-token-rows-untrained",
             "q3-1.7b-lora-rows-s0": "reference:special-token-rows-initialised", "q35-2b-lora-s1": "ablation:seed=1",
             "q35-2b-mixed-negatives": "ablation:negatives=200", "q35-2b-broken-rawtext": "ablation:raw-hermes-text"}[r["tag"]]
    body = {"label": label,
            "model": r["model"], "seed": r.get("seed"), **common, "env": RN["env"],
            "train": {k: r[k] for k in ("trainSec", "peakGB", "steps", "rows", "trainable", "loss") if k in r},
            "evals": {"summary": s, "genSec": r["genSec"], "ownVsBfclAgree": sum(a == b for a, b in zip(r["correct"], r["officialCorrect"])),
                      "errors": errors(r["outputs"], parse)},
            "outputs": r["outputs"], "tagCounts": extra}
    if "altParse" in r:
        body["evals"]["withJsonParser"] = r["altParse"]["summary"]
    if r["tag"] == "q35-2b-lora-s1":
        body["pairedBootstrap"] = {"s1VsBase": per_cat_ci(NB["base"]["correct"], r["correct"]), "s1VsS0": per_cat_ci(NB["sft"]["correct"], r["correct"])}
    if r["tag"] in ("q35-2b-mixed-negatives", "q35-2b-broken-rawtext"):
        body["pairedBootstrap"] = {"vsBase": per_cat_ci(NB["base"]["correct"], r["correct"]), "vsS0": per_cat_ci(NB["sft"]["correct"], r["correct"])}
    if r["tag"] == "q3-1.7b-lora-rows-s0":
        base3 = [x for x in RN["runs"] if x["tag"] == "q3-1.7b-base"][0]
        body["pairedBootstrap"] = {"vsBase": per_cat_ci(base3["correct"], r["correct"])}
    if r["tag"] == "q3-1.7b-lora-s0":
        base3 = [x for x in RN["runs"] if x["tag"] == "q3-1.7b-base"][0]
        body["pairedBootstrap"] = {"sftVsBase": per_cat_ci(base3["correct"], r["correct"]), "q35VsQ3": per_cat_ci(r["correct"], NB["sft"]["correct"])}
    if "rawRows" in RN and "broken" in r["tag"]:
        body["rawRows"] = RN["rawRows"]
    write(prefix + r["tag"].replace("broken-rawtext", "rawtext"), body)

# Try it data: every item with the base, LoRA and constrained replies, parsed calls and verdicts
tr = []
for i, it in enumerate(items):
    row = {"id": it["id"], "category": it["category"], "question": it["messages"][-1]["content"],
           "functions": [{"name": f["name"], "params": list(f["parameters"].get("properties", {})), "required": f["parameters"].get("required", [])} for f in it["functions"]],
           "expected": it["ground_truth"], "replies": {}}
    for key, label in [("base", "base"), ("sft", "lora"), ("baseForced", "baseConstrained"), ("sftForced", "loraConstrained")]:
        text = NB[key]["outputs"][i]
        row["replies"][label] = {"text": text, "calls": C.parse_xml_calls(text, it["tools"]), "correct": NB[key]["correct"][i]}
    tr.append(row)
json.dump({"source": prefix + "notebook-q35-2b-s0", "model": "Qwen/Qwen3.5-2B-Base", "summary": summ, "items": tr},
          open(tryit_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("tryit items", len(tr), os.path.getsize(tryit_path), "bytes")
print(json.dumps({k: v["accuracy"] for k, v in summ.items()}), own_vs_official)
