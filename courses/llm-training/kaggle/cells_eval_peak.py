"""Kaggle kernel llt-eval-peak (lltEvaluation audit pass, 2026-11-07). One T4, under 30 GPU-min.

Measures what the audit asked for, with the page cells unchanged:
- per-cell peak GPU memory (torch.cuda.max_memory_allocated, reset before each cell) for cells c01, c02, c06 and c07;
  c10's peak (9.17 GB) is already recorded by the cell itself in llt-eval-main v2;
- whether lm-eval's float16 log-softmax explains the gap between our scorer and the harness: rerun lm-eval
  arc_easy on the base model with softmax_dtype float32 and compare with our scorer on the same 1,000 items.
c06 needs FORMATS from c03 only for one printed comparison, so the recorded c03 output of llt-eval-main v2 is loaded
instead of rerunning c03 (8 min). c03's definitions (arc, LETTERS, gold_of, ...) are executed, its scoring run is not.
"""
import ast, json, os, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "labs", "lltEvaluation")
head = runpy.run_path(os.path.join(LAB, "notebook_head.py"))
read = lambda *p: open(os.path.join(LAB, *p), encoding="utf-8").read()
FORMATS_V2 = json.load(open(os.path.join(HERE, "out", "llt-eval-main", "results", "formats.json"), encoding="utf-8"))


def defs_only(src, keep):
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id in keep for t in n.targets))]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


SETUP = r'''
import json, time, traceback, torch, numpy as np
T0 = time.time()
PEAK = {"cells": {}, "errors": {}}
def save():
    json.dump(PEAK, open("peak_record.json", "w"), indent=1)
def run_cell(name, src):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        exec(compile(src, name, "exec"), globals())
    except Exception:
        PEAK["errors"][name] = traceback.format_exc()[-2500:]
        print("CELL FAILED", name, PEAK["errors"][name], flush=True)
    PEAK["cells"][name] = {"seconds": round(time.time() - t, 1), "peakAllocatedGB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           "peakReservedGB": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    print("--", name, PEAK["cells"][name], "| session", round((time.time() - T0) / 60, 1), "min", flush=True)
    save()
'''

SOFTMAX = r'''
lm_b = LMS[BASE]
lm_b.softmax_dtype = torch.float32                     # lm-eval's own option; default None = model dtype (float16 here)
out32 = lm_eval.simple_evaluate(model=lm_b, tasks=["arc_easy"], limit=LIMIT, log_samples=True, task_manager=TM)
rows32 = sorted(out32["samples"]["arc_easy"], key=lambda s: s["doc_id"])
theirs32 = np.array([float(r[0]) for s in rows32 for r in s["filtered_resps"]])
gap32 = np.abs(ours - theirs32)
acc32 = np.array([float(s["acc"]) for s in rows32])
PEAK["softmax"] = {"n": int(len(theirs32)), "fp16": {"maxGap": float(gap.max()), "meanGap": float(gap.mean()),
                   "itemsDiffer": int((acc != np.array(HARNESS[BASE]["tasks"]["arc_easy"]["acc"])).sum())},
                   "fp32": {"maxGap": float(gap32.max()), "meanGap": float(gap32.mean()), "itemsDiffer": int((acc != acc32).sum()),
                            "harnessAcc": float(acc32.mean())},
                   "pickDiffers": int(pick_differs)}
print("softmax fp16 vs ours:", PEAK["softmax"]["fp16"], "| fp32 vs ours:", PEAK["softmax"]["fp32"], flush=True)
lm_b.softmax_dtype = None
save()
'''

INJECT = ("FORMATS = {m: {f: np.array(v) for f, v in d.items()} for m, d in json.loads(" + repr(json.dumps(FORMATS_V2)) + ").items()}\n"
          + defs_only(read("cells", "c03_formats.py"), ("LETTERS", "arc")) + "\n")

GUARD = r'''
if (time.time() - T0) / 60 < 21:
    run_cell("c07_contamination", open("cells/c07_contamination.py", encoding="utf-8").read())
else:
    PEAK["skipped"] = "c07, time guard at 21 min"
PEAK["totalMinutes"] = round((time.time() - T0) / 60, 1)
save()
print("DONE", PEAK["totalMinutes"], flush=True)
'''

files = {f: read("cells", f) for f in ["c01_harness.py", "c02_scorer.py", "c06_generative.py", "c07_contamination.py"]}
WRITE = "import os\nos.makedirs('cells', exist_ok=True)\n" + "".join(
    "open(" + repr("cells/" + k) + ", 'w', encoding='utf-8').write(" + json.dumps(v) + ")\n" for k, v in files.items())

CELLS = [("md", "# lltEvaluation audit pass: per-cell peak memory and the float32 softmax check"),
         ("code", head["RUNTIME_CHECK"]), ("code", head["INSTALL"]), ("code", WRITE + SETUP),
         ("code", "run_cell('c01_harness', open('cells/c01_harness.py', encoding='utf-8').read())"),
         ("code", "run_cell('c02_scorer', open('cells/c02_scorer.py', encoding='utf-8').read())"),
         ("code", "run_cell('softmax_fp32_check', " + repr(SOFTMAX) + ")"),
         ("code", "run_cell('c03_defs_and_recorded_formats', " + repr(INJECT) + ")"),
         ("code", "run_cell('c06_generative', open('cells/c06_generative.py', encoding='utf-8').read())"),
         ("code", GUARD)]
