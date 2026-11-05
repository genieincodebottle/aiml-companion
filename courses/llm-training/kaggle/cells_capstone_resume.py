"""Kaggle kernel llt-capstone-gpt2-resume: exact-resume check of the reference train.py on 2x T4, a branch decay
(the learner path), and the lm-eval wrapper on a short limit.
Build: python build_nb.py cells_capstone_resume.py llt-capstone-gpt2-resume llt-capstone-gpt2-resume "LLT Capstone GPT2 Resume"
"""
import os, runpy

V1 = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_capstone_gpt2.py"))
SETUP = V1["SETUP"].replace("lltCapstoneGPT2_measure.json", "lltCapstoneGPT2_resume.json")
write_cell = V1["write_cell"]

COMMON = r'''
D = "/kaggle/working/data"
ARGS = ["--data", D, "--micro", "8", "--tokens-per-step", "65536", "--max-steps", "12", "--warmup", "2",
        "--stable-end", "8", "--save-every", "6", "--milestones", "6,12", "--export-every", "12",
        "--val-every", "6", "--val-tokens", "131072", "--sample-every", "12"]
TR = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "train.py"]
out = sh("prep", [sys.executable, "prep_fineweb.py", "--out", D, "--files", "2", "--max-docs", "1000",
                  "--val-tokens", "300000", "--workers", "2"], timeout=400)
RUN["data"] = result_line(out)
print(json.load(open(D + "/index.json"))["train"])
'''

RUN_A = r'''
sh("runA", TR + ARGS + ["--out", "/kaggle/working/A", "--run-id", "resumeA"], timeout=600)
ck = "/kaggle/working/A/ckpt/step_000006"
RUN["ckptFiles"] = {f: os.path.getsize(os.path.join(ck, f)) for f in sorted(os.listdir(ck))}
RUN["ckptGB"] = round(sum(RUN["ckptFiles"].values()) / 1e9, 3)
print(RUN["ckptFiles"], RUN["ckptGB"])
save()
'''

RUN_B = r'''
sh("runB", TR + ARGS + ["--out", "/kaggle/working/B", "--run-id", "resumeB", "--resume", "/kaggle/working/A/ckpt/step_000006"],
   timeout=600)
A = json.load(open("/kaggle/working/A/resumeA.json"))["metrics"]
B = json.load(open("/kaggle/working/B/resumeB.json"))["metrics"]
cmp = []
for ma, mb in zip(A[6:], [m for m in B if m["step"] > 6]):
    row = {"step": ma["step"], "shardA": ma["loaderShard"], "shardB": mb["loaderShard"]}
    for k in ("trainLoss", "gradNorm", "valLoss", "lossScale", "lr"):
        if k in ma:
            row[k] = [ma[k], mb.get(k)]
    cmp.append(row)
RUN["resumeCompare"] = cmp
diffs = [abs(r["trainLoss"][0] - r["trainLoss"][1]) for r in cmp]
RUN["resumeMaxAbsDiffTrainLoss"] = max(diffs)
RUN["resumeBitwiseTrainLoss"] = all(d == 0 for d in diffs)
for r in cmp:
    print(r)
print("max abs diff", max(diffs), "bitwise", RUN["resumeBitwiseTrainLoss"])
save()
'''

RUN_C = r'''
if time.time() - T_START < 9 * 60:
    sh("runC_branch", TR + ARGS[:6] + ["--max-steps", "9", "--warmup", "2", "--stable-end", "6", "--save-every", "9",
       "--milestones", "9", "--export-every", "9", "--val-every", "3", "--val-tokens", "131072", "--sample-every", "9",
       "--out", "/kaggle/working/C", "--run-id", "branchC", "--resume", "/kaggle/working/A/ckpt/step_000006"], timeout=600)
    C = json.load(open("/kaggle/working/C/branchC.json"))["metrics"]
    RUN["branch"] = [{k: m.get(k) for k in ("step", "lr", "trainLoss", "valLoss")} for m in C if m["step"] > 6]
    print(RUN["branch"])
    save()
'''

EVAL = r'''
if time.time() - T_START < 11 * 60:
    sh("pip_lmeval", [sys.executable, "-m", "pip", "install", "-q", "lm-eval==0.4.13"], timeout=300)
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    out = sh("eval_wrapper", [sys.executable, "eval.py", "--weights", "/kaggle/working/A/weights/weights_step_000012.pt",
                              "--val", D + "/val.bin", "--limit", "50", "--batch-size", "32",
                              "--out", "/kaggle/working/evals"], timeout=600, env=env1)
import shutil
for d in ("/kaggle/working/data", "/kaggle/working/A/ckpt", "/kaggle/working/B/ckpt", "/kaggle/working/C/ckpt",
          "/kaggle/working/A/weights", "/kaggle/working/B/weights", "/kaggle/working/C/weights"):
    shutil.rmtree(d, ignore_errors=True)
print("total min", round((time.time() - T_START) / 60, 1))
save()
'''

CELLS = [
    ("md", "# lltCapstoneGPT2 exact-resume check (2x T4)\n\nRuns the H100 reference script on a small budget: "
           "12 steps straight, then a fresh process resumed from step 6, then a branch decay from step 6."),
    ("code", SETUP),
    write_cell("gpt.py"), write_cell("prep_fineweb.py"), write_cell("train.py"), write_cell("hf_wrapper.py"),
    write_cell("eval.py"),
    ("code", COMMON), ("code", RUN_A), ("code", RUN_B), ("code", RUN_C), ("code", EVAL),
]
