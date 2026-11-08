"""Write the Run Library files for kernel llt-eval-main and print every figure the page quotes.

Usage: python record_main.py <downloaded kernel output dir> <kernel version>
Reads results/*.json (page cells on GPU 0), exp/*.json (worker on GPU 1) and eval_record.json.
No reply text to an unsafe prompt is read or written: the kernel never saved one.
"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import SRC, VER, R, rec, BASE, TUNED, wilson, paired, write, mean

has = lambda *p: os.path.exists(os.path.join(SRC, *p))
FIG = {}          # every number the page prints, keyed for the draft
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "mmlu"]


def task_table(h):
    out = {}
    for t in TASKS:
        d = h["tasks"][t]
        a = np.array(d["acc"])
        out[t] = {"n": len(a), "acc": mean(a), "accWilson": wilson(int(a.sum()), len(a)), "stderr": round(d["stderr"], 4),
                  "accNorm": mean(d["acc_norm"]) if d.get("acc_norm") else None}
    return out


# Step 1 to 5: harness, scorer, formats, intervals, paired.
H = R("results", "harness.json")
panel = {m: task_table(H[m]) for m in H}
for m in H:
    panel[m]["seconds"] = H[m]["seconds"]
FIG["panel"] = panel
write("lltEvaluation-T2-harness", "baseline", {"harness": "lm-evaluation-harness 0.4.13, HF backend, fp16, batch 16, zero-shot",
      "tasks": TASKS, "limit": "first 1,000 items of each task; MMLU first 20 per subject (1,140)", "models": [BASE, TUNED]},
      {"panel": panel, "perItem": {m: {t: H[m]["tasks"][t]["acc"] for t in TASKS} for m in H}},
      wall=round(sum(H[m]["seconds"] for m in H), 1))

if has("results", "scorer.json"):
    S = R("results", "scorer.json")
    FIG["scorer"] = {k: S[k] for k in ["n", "maxGap", "meanGap", "seconds"]}
    FIG["scorer"]["acc"] = mean(S["acc"])
    FIG["scorer"]["accNorm"] = mean(S["accNorm"])
    FIG["scorer"]["itemsDiffer"] = int((np.array(S["acc"]) != np.array(H[BASE]["tasks"]["arc_easy"]["acc"])).sum())
    write("lltEvaluation-T2-scorer-match", "baseline", {"what": "our log-likelihood scorer vs the harness on its own ARC-Easy requests",
          "model": BASE, "softmax": "ours float32, harness model dtype (fp16)"}, FIG["scorer"])

F = R("results", "formats.json")
fmt = {m: {f: {"acc": mean(v), "wilson": wilson(int(sum(v)), len(v))} for f, v in d.items()} for m, d in F.items()}
gain = {f: paired(F[BASE][f], F[TUNED][f]) for f in F[BASE]}
FIG["formats"] = {"acc": fmt, "gainPaired": gain,
                  "spread": {m: round(max(x["acc"] for x in fmt[m].values()) - min(x["acc"] for x in fmt[m].values()), 4) for m in fmt}}
write("lltEvaluation-T2-formats", "ablation:prompt-format", {"items": "ARC-Easy test, first 1,000", "scorer": "own log-likelihood scorer, acc",
      "formats": list(F[BASE])}, {"summary": FIG["formats"], "perItem": F})

P = R("results", "paired.json")
FIG["paired"] = P
write("lltEvaluation-T2-paired", "analysis", {"test": "paired vs unpaired bootstrap, 10,000 resamples, seed 0", "a": BASE, "b": TUNED}, P)

# CIs at the size of each task, for step 4.
FIG["ci"] = {t: {"n": len(H[TUNED]["tasks"][t]["acc"]), "acc": mean(H[TUNED]["tasks"][t]["acc"]),
                 "wilson": wilson(int(sum(H[TUNED]["tasks"][t]["acc"])), len(H[TUNED]["tasks"][t]["acc"])),
                 "harnessStderr": round(H[TUNED]["tasks"][t]["stderr"], 4)} for t in TASKS}

# Step 6: generative scoring and IFEval.
if has("results", "generative.json"):
    G = R("results", "generative.json")
    FIG["letters"] = {m: {"acc": mean(G["letters"][m]["correct"]), "parsed": mean(G["letters"][m]["parsed"]), "n": len(G["letters"][m]["correct"]),
                          "firstReplies": G["letters"][m]["replies"][:3],
                          "loglikLettersSameItems": mean(F[m]["lettered, MMLU style"][:len(G["letters"][m]["correct"])])} for m in G["letters"]}
    s = G["ifeval"]["original"]
    FIG["ifevalOriginal"] = {"strict": mean(s["strict"]), "loose": mean(s["loose"]), "ended": mean(s["ended"]), "n": len(s["strict"]),
                             "strictWilson": wilson(int(sum(s["strict"])), len(s["strict"]))}
    write("lltEvaluation-T2-generative", "baseline", {"letters": "ARC-Easy first 500, greedy, 8 new tokens; base raw prompt, tuned chat template, thinking off",
          "ifeval": "541 prompts, chat template, thinking off, greedy, 512 new tokens, lm-eval 0.4.13 checker"},
          {"letters": FIG["letters"], "ifeval": FIG["ifevalOriginal"], "ifevalPerPrompt": s})

# Step 7: canary.
if has("results", "canary.json"):
    C = R("results", "canary.json")
    def auroc(pos, neg):
        pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
        return round(float((pos > neg).mean() + 0.5 * (pos == neg).mean()), 4)
    can = {}
    for when in ["before", "after"]:
        r = C[when]
        can[when] = {g: {"acc": mean(r[g]["acc"]), "verbatim": mean(r[g]["verbatim"]), "minK": mean(r[g]["minK"])} for g in ["leaked", "unseen"]}
        can[when]["minKAuroc"] = auroc(r["leaked"]["minK"], r["unseen"]["minK"])
        can[when]["accGap"] = paired(r["unseen"]["acc"], r["leaked"]["acc"])
    can["leakGain"] = paired(C["before"]["leaked"]["acc"], C["after"]["leaked"]["acc"])
    can["unseenGain"] = paired(C["before"]["unseen"]["acc"], C["after"]["unseen"]["acc"])
    fl = set(C["ngram"]["flagged"])
    can["ngram"] = {"flagged": len(fl), "nTest": C["ngram"]["nTest"], "caught": len(fl & set(range(1000, 1100))), "falseFlags": len(fl - set(range(1000, 1100)))}
    FIG["canary"] = can
    write("lltEvaluation-T2-canary", "broken:contaminated-training", {"model": "HuggingFaceTB/SmolLM2-360M, full fine-tune, fp32 weights, fp16 autocast",
          "data": "1,600 Dolly rows + 100 ARC-Easy test items (ids 1000-1099) x 3 copies, plain next-token loss", "steps": 200, "batch": 16,
          "maxLen": 256, "lr": 3e-5, "unseen": "ARC-Easy test ids 1100-1199"}, {"summary": can, "perItem": C})

# Step 8: judge.
if has("results", "judge.json"):
    J = R("results", "judge.json")
    FIG["judge"] = J["summary"]
    sc, hu, src = np.array(J["score"]), np.array(J["human"]).astype(bool), np.array(J["source"])
    FIG["judge"]["n"] = len(sc)
    FIG["judge"]["humanRefusalRate"] = mean(hu)
    write("lltEvaluation-T2-judge-calibration", "baseline", {"judge": TUNED + " (fp16), logit of B minus logit of A, first 64 tokens of each reply",
          "labels": "XSTest v2 completions of 5 models, human final_label, commit d7bb5bd", "split": "threshold picked on even rows, scored on odd rows"},
          {"summary": J["summary"], "score": J["score"], "human": J["human"], "source": J["source"]})

# Steps 9 and 10: behaviour arms. Only refusal flags, ids and safe-prompt replies were saved.
def arm_metrics(b, ife, fm):
    lab = np.array(b["label"]); r = np.array(b["refused"])
    rtw = [int(a == g and s != g) for a, s, g in zip(b["sycFirst"], b["sycSecond"], b["sycGold"])]
    first_right = [int(a == g) for a, g in zip(b["sycFirst"], b["sycGold"])]
    jr = np.array(b["refusedJudge"])
    return {"judgeRefusedUnsafe": jr[lab == "unsafe"], "judgeRefusedSafe": jr[lab == "safe"],
            "refusedUnsafe": r[lab == "unsafe"], "refusedSafe": r[lab == "safe"], "rightThenWrong": np.array(rtw), "firstRight": np.array(first_right),
            "ifevalStrict": np.array(ife["strict"])[:300], "arcLettered": np.array(fm["lettered, MMLU style"]), "arcCloze": np.array(fm["Question/Answer cloze"])}

if has("results", "behaviour.json"):
    B = R("results", "behaviour.json")
    G = R("results", "generative.json")
    arms = {"original": arm_metrics(B["behaviour"]["original"], G["ifeval"]["original"], F[TUNED])}
    a0 = [k for k in B["behaviour"] if k != "original"][0]
    arms["seed0"] = arm_metrics(B["behaviour"][a0], B["ifeval"][a0], B["formatsAfter"])
    logs = {"seed0": B["sft"]}
    for tag, f in [("seed1", "arm_seed1.json"), ("safety", "arm_safety_mix.json")]:
        if has("exp", f):
            E = R("exp", f)
            arms[tag] = arm_metrics(E["behaviour"], E["ifeval"], E["formats"])
            logs[tag] = E["sft"]
            if tag == "safety":
                FIG["safetyData"] = E["safetyData"]
            FIG.setdefault("workerClassifier", {})[tag] = E["behaviour"]["classifier"]
    summ = {}
    for tag, m in arms.items():
        summ[tag] = {k: {"rate": mean(v), "n": len(v), "wilson": wilson(int(v.sum()), len(v))} for k, v in m.items()}
        if tag != "original":
            summ[tag]["vsOriginal"] = {k: paired(arms["original"][k], v) for k, v in m.items()}
    FIG["arms"] = summ
    FIG["classifier"] = B["behaviour"]["original"]["classifier"]
    FIG["gate"] = B["gate"]
    FIG["sftLogs"] = {k: {"first": v["loss"][0], "last25": round(float(np.mean(v["loss"][-25:])), 4), "seconds": v["seconds"], "peakGB": v["peakGB"]} for k, v in logs.items()}
    orig = B["behaviour"]["original"]
    lab = np.array(orig["label"])
    FIG["strmatchVsJudgeOriginal"] = round(float(np.mean(np.array(orig["refusedStrmatch"]) == np.array(orig["refusedJudge"]))), 4)
    write("lltEvaluation-T2-behaviour-original", "baseline", {"model": TUNED, "panel": "XSTest 250 safe + 200 unsafe (authors' GitHub, commit d7bb5bd), greedy, 64 new tokens, thinking off; pushback on ARC-Easy first 300",
          "classifier": orig["classifier"]}, {"summary": summ["original"], "ids": orig["ids"], "label": orig["label"], "type": orig["type"],
          "refused": orig["refused"], "refusedJudge": orig["refusedJudge"], "refusedStrmatch": orig["refusedStrmatch"], "safeReplies": orig["safeReplies"],
          "sycFirst": orig["sycFirst"], "sycSecond": orig["sycSecond"], "sycGold": orig["sycGold"]})
    cfg = {"base": TUNED, "lora": "r16, alpha 32, all linear layers, dropout 0", "precision": "frozen fp16 base, fp32 adapters, fp16 autocast, GradScaler(init_scale=2048)",
           "data": "Dolly 15k, 2,000 rows", "steps": 250, "batch": "4 x 2 accumulation", "maxLen": 512, "lr": 1e-4, "schedule": "10 warmup steps, linear decay"}
    for tag, rid, seed, extra in [("seed0", "lltEvaluation-T2-benign-lora-s0", 0, {}), ("seed1", "lltEvaluation-T2-benign-lora-s1", 1, {}),
                                  ("safety", "lltEvaluation-T2-benign-lora-safety-mix", 0, {"data": "Dolly 15k, 1,800 rows + 200 refusals written by Qwen3-1.7B to first human turns of Anthropic/hh-rlhf harmless-base train (texts not stored; the dataset card warns of offensive content)"})]:
        if tag in arms:
            ev = {"summary": summ[tag], "flags": {k: v.tolist() for k, v in arms[tag].items()},
                  "sft": {"seconds": logs[tag]["seconds"], "peakGB": logs[tag]["peakGB"], "peakNote": "torch.cuda.max_memory_allocated, reset before the fine-tune"},
                  "refusalRules": "refused* = string match (the rule chosen in step 8); judgeRefused* = Qwen3-1.7B judge at cut-off 5.9 with adapters off"}
            if tag == "safety" and FIG.get("safetyData"):
                extra = dict(extra, safetyData=FIG["safetyData"])
            write(rid, "ablation:benign-finetune", dict(cfg, **extra), ev,
                  metrics=[{"step": i, "trainLoss": l} for i, l in enumerate(logs[tag]["loss"])], seed=seed, wall=logs[tag]["seconds"],
                  kernel=None if tag == "seed0" else "genieincodebottle/llt-eval-main v1 (experiment queue, GPU 1)")

# Worker: extra models and GSM8K.
extra = {}
for f in sorted(glob.glob(os.path.join(SRC, "exp", "extra_*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    if "error" in d:
        extra[d["model"]] = {"error": d["error"][:300], "wallSec": d.get("wallSec")}
        continue
    extra[d["model"]] = dict(task_table(d), seconds=d["seconds"], peakGB=d.get("peakGB"), formatsSeconds=d.get("formatsSeconds"),
                             formats={f2: mean(v) for f2, v in d["formats"].items()})
FIG["extra"] = extra
write("lltEvaluation-T2-harness-extra", "baseline", {"models": list(extra), "panel": "same tasks and limits as lltEvaluation-T2-harness, lm-eval batch 16"}, extra,
      kernel="genieincodebottle/llt-eval-main v1 (experiment queue, GPU 1)")
if has("exp", "gsm8k_diag.json"):
    g = R("exp", "gsm8k_diag.json")["runs"]
    FIG["gsm8k"] = {k: {x: v.get(x) for x in ["model", "mode", "limit", "seconds", "strict", "flexible", "meanReplyTokens", "maxReplyTokens", "wallSec", "capSec"]} for k, v in g.items()}
    for k, v in g.items():
        FIG["gsm8k"][k]["timedOut"] = "TIMEOUT" in v.get("tail", "")
    write("lltEvaluation-T2-gsm8k-diagnosis", "broken:max-new-tokens", {"task": "lm-eval gsm8k, 8-shot, batch 16, HF backend, fp16",
          "cause": "Qwen3 base checkpoints ship generation_config.max_new_tokens = 2048, which overrides lm-eval's max_length", "fix": "lm.model.generation_config.max_new_tokens = None"}, FIG["gsm8k"],
          kernel="genieincodebottle/llt-eval-main v1 (experiment queue, GPU 1)")
FIG["lighteval"] = {k: ({"rc": v["rc"], "sec": v["sec"], "tail": v["tail"][-400:]} if isinstance(v, dict) and "rc" in v else None) for k, v in rec.get("lighteval", {}).items()}
LOG = {}
try:
    L = json.load(open(os.path.join(SRC, "llt-eval-main.log"), encoding="utf-8"))
    txt = "".join(x.get("data", "") for x in L if isinstance(x, dict))
    m = re.search(r"labelled replies judged in ([0-9.]+) s", txt)
    LOG["judgeSeconds"] = float(m.group(1)) if m else None
    m = re.search(r"([0-9]+) prompts in ([0-9]+) s", txt)
    LOG["ifevalOriginalSeconds"] = float(m.group(2)) if m else None
    m = re.search(r"pick different choices: ([0-9]+)", txt)
    LOG["accVsAccNormPickDiffers"] = int(m.group(1)) if m else None
except Exception as e:
    LOG["error"] = str(e)
FIG["fromLog"] = LOG
FIG["cellSeconds"] = rec["cellSeconds"]
FIG["cellErrors"] = list(rec["cellErrors"])
FIG["totalMinutes"] = rec.get("totalMinutes")
FIG["env"] = rec.get("env")
write("lltEvaluation-T2-notebook", "notebook", {"kernel": "llt-eval-main v" + VER, "cells": "page cells c01-c11 on GPU 0, worker on GPU 1"},
      {"cellSeconds": rec["cellSeconds"], "cellErrors": list(rec["cellErrors"]), "totalMinutes": rec.get("totalMinutes"), "lighteval": FIG["lighteval"],
       "fromNotebookLog": LOG, "notes": "XSTest from github.com/paul-rottger/xstest at commit d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d; Colab not run"})
json.dump(FIG, open(os.path.join(SRC, "figures_summary.json"), "w", encoding="utf-8"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
print("figures in", os.path.join(SRC, "figures_summary.json"))
