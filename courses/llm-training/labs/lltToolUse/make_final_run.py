"""Write runs/lltToolUse/lltToolUse-T2-notebook-final.json from the llt-tooluse-final kernel outputs.

Usage: python make_final_run.py <out dir of llt-tooluse-final>   (BFCL_DIR env var: folder holding an unpacked bfcl_eval)
Re-scores every recorded reply with tool_common, checks the verdicts equal the recorded ones, and adds the log
timestamps of each step so the page can quote wall-clock figures from this run.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ["BFCL_DIR"])
import tool_common as C

out = sys.argv[1]
PLAN = {"simple_python": 4, "multiple": 4, "parallel": 4, "parallel_multiple": 4, "irrelevance": 4}
items = C.load_bfcl(os.path.join(os.environ["BFCL_DIR"], "bfcl_eval", "data"), PLAN)
NB = json.load(open(os.path.join(out, "notebook_results.json"), encoding="utf-8"))
EX = json.load(open(os.path.join(out, "notebook_final_exp.json"), encoding="utf-8"))
AD = json.load(open(os.path.join(out, "notebook_final_adapter.json"), encoding="utf-8"))
log = json.load(open(os.path.join(out, "llt-tooluse-final.log"), encoding="utf-8"))


def rescore(outs, parse, recorded):
    rows = C.score(items, outs, parse)
    assert [r["correct"] for r in rows] == recorded, "re-score differs"
    return C.summary(rows)


def first_time(marker):
    for e in log:
        if marker in e.get("data", ""):
            return round(e["time"], 1)
    return None


marks = {"step1": "the tool prompt adds", "step2": "training conversations |", "step3check": "labels equal", "step4": "310 items",
         "step6": "Qwen3 JSON | right answers", "step7": "agreement with BFCL", "step8train": "steps 300", "step8": "gain and 95%",
         "step9": "LoRA, constrained", "step10norms": "median row norm", "step10base": "Qwen3-1.7B-Base | relevant",
         "step10lora": "LoRA | relevant", "step10rows": "LoRA, special rows set first | relevant", "step11trained": "trained in",
         "step11": "change against the base", "adapter": "adapter pass 0", "end": "All checks passed"}
body = {"runId": "lltToolUse-T2-notebook-final", "module": "lltToolUse", "tier": "T2", "date": "2026-11-08",
        "label": "baseline: final learner notebook end to end, steps 10 and 11 on, then step 8 with USE_OUR_ADAPTER = True",
        "hardware": {"gpu": NB["env"]["gpu"], "count": 1, "vramGB": 15.6, "where": "Kaggle T4, kernel llt-tooluse-final v1"},
        "env": NB["env"], "check": NB["check"], "train": NB.get("train"),
        "evals": {k: {"summary": rescore(NB[k]["outputs"], C.parse_xml_calls, NB[k]["correct"]), "genSec": NB[k]["genSec"],
                      "ownVsBfclAgree": sum(a == b for a, b in zip(NB[k]["correct"], NB[k]["officialCorrect"]))}
                  for k in ("base", "sft", "baseForced", "sftForced")},
        "forcedSec": NB["forcedSec"], "pairedBootstrap": NB["pairedBootstrap"],
        "step10": {k: {"summary": rescore(v["outputs"], C.parse_json_calls, v["correct"]), "tagReplies": v["tagReplies"]} for k, v in EX["step10"].items()},
        "step11": {"summary": rescore(EX["step11"]["outputs"], C.parse_xml_calls, EX["step11"]["correct"]),
                   "pairedBootstrapVsBase": EX["step11"]["pairedBootstrapVsBase"]},
        "adapterPass": {"summary": rescore(AD["outputs"], C.parse_xml_calls, AD["correct"]), "genSec": AD["genSec"],
                        "pairedBootstrapVsBase": AD["pairedBootstrapVsBase"]},
        "logSeconds": {k: first_time(m) for k, m in marks.items()},
        "outputs": {"base": NB["base"]["outputs"], "sft": NB["sft"]["outputs"], "step11": EX["step11"]["outputs"], "adapter": AD["outputs"],
                    **{"step10:" + k: v["outputs"] for k, v in EX["step10"].items()}}}
json.dump(body, open(os.path.join(HERE, "..", "..", "runs", "lltToolUse", "lltToolUse-T2-notebook-final.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps({k: body[k] for k in ("check", "train", "forcedSec", "pairedBootstrap", "logSeconds")}, default=str)[:3000])
for k, v in body["evals"].items():
    print(k, v["summary"]["accuracy"], {c: x["accuracy"] for c, x in v["summary"]["byCategory"].items()}, v["genSec"], v["ownVsBfclAgree"])
for k, v in body["step10"].items():
    print("step10", k, v["summary"]["accuracy"], v["tagReplies"])
print("step11", body["step11"]["summary"]["accuracy"], {c: x["accuracy"] for c, x in body["step11"]["summary"]["byCategory"].items()}, body["step11"]["pairedBootstrapVsBase"])
print("adapter", body["adapterPass"]["summary"]["accuracy"], {c: x["accuracy"] for c, x in body["adapterPass"]["summary"]["byCategory"].items()}, body["adapterPass"]["pairedBootstrapVsBase"])
