# make_ref_runs.py - turn the reference run's deliverables (Google Cloud, llt-c4) into Run Library files.
#
#   python make_ref_runs.py --src <extracted deliverables.tar.gz> --quick <extracted lcheck.tgz>
#
# Writes courses/llm-training/runs/lltCapstoneGPT2/: the main run and branch records (with the layout change and the
# data wrap written into the record), every eval file, the data indexes, a compute record (FLOPs, MFU, GPU-hours per
# stage from the VM's boot records and file times), the learner quick-path check, and filtered logs.
import argparse, datetime as dt, glob, json, os, re, shutil, statistics as st

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)            # holds run/, branches/, data/index.json
p.add_argument("--quick", default="")             # holds decay/*.json, evals/, ldata_index.json, status, *.log
p.add_argument("--timeline", default="")          # JSON with the VM times (see TIMELINE below); overrides the defaults
a = p.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "runs", "lltCapstoneGPT2"))
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
S = a.src
TPS = 524_288
PEAK = 312e12                                       # A100 dense bf16, the peak train.py uses for MFU

# VM times, UTC, 2026-11-10: `last -x` (boots and shutdowns), torchrun warnings in train.log, file times on the VM.
TIMELINE = {
    "boot1GPU": "2026-11-10T08:42:38Z", "shutdown1GPU": "2026-11-10T10:51:55Z",     # a2-highgpu-1g, 1 x A100 40GB
    "prepEnd": "2026-11-10T09:12:55Z", "calibrationEnd": "2026-11-10T09:27:27Z",
    "mainStart": "2026-11-10T09:29:30Z",                                              # ledger start line
    "boot4GPU": "2026-11-10T10:53:25Z", "resume4GPU": "2026-11-10T10:54:19Z",         # a2-highgpu-4g, 4 x A100 40GB
    "mainEnd": "2026-11-10T14:44:52Z", "branchesEnd": "2026-11-10T15:10:33Z",
    "evalRelaunch1": "2026-11-10T18:29:22Z", "evalRelaunch2": "2026-11-10T18:40:55Z",
    "evalsEnd": None, "quickStart": "2026-11-10T18:37:30Z", "quickEnd": None,
}
if a.timeline:
    TIMELINE.update(json.load(open(a.timeline)))


def t(key):
    return dt.datetime.strptime(TIMELINE[key], "%Y-%m-%dT%H:%M:%SZ")


def secs(k0, k1):
    return round((t(k1) - t(k0)).total_seconds(), 1)


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1)
    return name


written = []
# ---- main run ----------------------------------------------------------------------------------------------------
main = json.load(open(os.path.join(S, "run", "lltCapstoneGPT2-ref-124m-10B.json")))
m = main["metrics"]
log = open(os.path.join(S, "run", "train.log"), errors="replace").read().splitlines()
resumes = [l for l in log if l.startswith("resumed from")]
heads = [l for l in log if l.startswith("124m params")]
# The 1-GPU process logged steps past its last checkpoint (step 1,500) before the VM was stopped; those were rerun on 4 GPUs.
first_resume = next(i for i, l in enumerate(log) if l.startswith("resumed from"))
lost = [json.loads(l) for l in log[:first_resume] if l.startswith('{"step"') and json.loads(l)["step"] > 1500]


def seg(lo, hi, gpus):
    mm = [x for x in m if lo <= x["step"] <= hi and x["step"] - lo >= 10]   # skip the compile steps after each start
    return {"gpus": gpus, "gpu": "NVIDIA A100-SXM4-40GB", "steps": [lo, hi], "layout": f"micro 32 x accum {16 // gpus}",
            "medianTokPerSec": st.median(x["tokPerSec"] for x in mm), "medianMfu": st.median(x["mfu"] for x in mm),
            "peakMemGB": max(x["memGB"] for x in m if lo <= x["step"] <= hi),
            "trainSec": round(m[hi - 1]["wallSec"] - (m[lo - 2]["wallSec"] if lo > 1 else 0), 1)}


wrap = next(x for x in m if x["epoch"] == 1)
data_index = json.load(open(os.path.join(S, "data", "index.json")))
main["segments"] = [seg(1, 1500, 1), seg(1501, 19073, 4)]
main["layoutLog"] = {"resumedLines": resumes, "startLines": heads,
                     "note": "The record's hardware field shows the last process (4 GPUs). Steps 1-1,500 ran on 1 GPU; "
                             "the VM was then resized and training resumed from the step 1,500 checkpoint on 4 GPUs."}
main["discardedSteps"] = {"gpus": 1, "fromStep": 1501, "lastLoggedStep": lost[-1]["step"] if lost else None,
                          "lastLoggedWallSec": lost[-1]["wallSec"] if lost else None,
                          "note": "Trained on 1 GPU after the last checkpoint, lost when the VM was stopped for the resize, "
                                  "then trained again on 4 GPUs. The metrics array holds the 4-GPU values."}
main["dataWrap"] = {"step": wrap["step"], "trainTokensAvailable": data_index["trainTokens"],
                    "tokensRequested": m[-1]["tokens"],
                    "tokensRereadAtLeast": m[-1]["tokens"] - data_index["trainTokens"],
                    "note": "The loader reached the end of the 14 files and restarted at file 0 (epoch 1)."}
main.pop("costUSD", None)
main.pop("priceCheckedOn", None)
written.append(dump("lltCapstoneGPT2-ref-124m-10B.json", main))

# ---- branch decays -----------------------------------------------------------------------------------------------
wall_at = {x["step"]: x["wallSec"] for x in m}
branches = {}
for f in sorted(glob.glob(os.path.join(S, "branches", "*", "lltCapstoneGPT2-ref-124m-*.json"))):
    b = json.load(open(f))
    start = b["resumes"][0]["step"]
    b["branchOf"] = {"runId": main["runId"], "fromStep": start, "stableEnd": b["config"]["stable_end"],
                     "maxSteps": b["config"]["max_steps"]}
    b["branchTrainSec"] = round(b["metrics"][-1]["wallSec"] - wall_at[start], 1)   # wallSec carries the main run's time
    b.pop("costUSD", None)
    b.pop("priceCheckedOn", None)
    branches[b["runId"]] = b
    written.append(dump(b["runId"] + ".json", b))

# ---- evals -------------------------------------------------------------------------------------------------------
E = os.path.join(S, "run", "evals")
for f in sorted(glob.glob(os.path.join(E, "**", "*.json"), recursive=True)):
    rel = os.path.relpath(f, E)
    os.makedirs(os.path.join(OUT, "evals", os.path.dirname(rel)), exist_ok=True)
    shutil.copyfile(f, os.path.join(OUT, "evals", rel))
    written.append("evals/" + rel.replace(os.sep, "/"))

# ---- data index, calibration -------------------------------------------------------------------------------------
written.append(dump("lltCapstoneGPT2-ref-data-index.json", {"runId": "lltCapstoneGPT2-ref-data-index", **data_index}))
cal_path = os.path.join(S, "run", "calibration.json")
if os.path.exists(cal_path):
    cal = json.load(open(cal_path))
    cal = {k: v for k, v in cal.items() if "USD" not in k and "price" not in k.lower()}
    written.append(dump("lltCapstoneGPT2-ref-calibration.json", {"runId": "lltCapstoneGPT2-ref-calibration", **cal}))

# ---- logs (progress bars removed) --------------------------------------------------------------------------------
for src, name in [("run/train.log", "train.log"), ("run/prep.log", "prep.log"), ("run/eval.log", "eval.log"),
                  ("run/nvidia-smi.txt", "nvidia-smi.txt"), ("run/pip-freeze.txt", "pip-freeze.txt")] + \
        [(os.path.relpath(g, S), "branch-" + os.path.basename(g)) for g in glob.glob(os.path.join(S, "branches", "*.log"))]:
    path = os.path.join(S, src)
    if not os.path.exists(path):
        continue
    lines = []
    for line in open(path, errors="replace").read().replace("\r", "\n").splitlines():
        if re.search(r"\d+%\|", line) or not line.strip():
            continue
        lines.append(line)
    with open(os.path.join(OUT, "logs", name), "w") as f:
        f.write("\n".join(lines) + "\n")
    written.append("logs/" + name)

# ---- learner quick-path check (fp16, 2 ranks, micro 8, on 2 A100s standing in for 2 T4s) ----------------------------
if a.quick:
    q = json.load(open(glob.glob(os.path.join(a.quick, "decay", "*.json"))[0]))
    q["label"] = "check:learner-quick-path"
    q["tier"] = "T4 numerics check"
    q["note"] = ("The learner quick path run with the T4 settings (fp16 + GradScaler, 2 ranks, micro 8, data files 0, 12, 13 "
                 "with 150,000 documents each, --reset-loader-shard 1) on 2 idle A100s of the same VM. Times are A100 times, "
                 "not T4 times. The starting state was the local step 4,529 checkpoint, the same file uploaded to the Hub.")
    q.pop("costUSD", None)
    q.pop("priceCheckedOn", None)
    written.append(dump("lltCapstoneGPT2-quickpath-fp16-check.json", q))
    for f in sorted(glob.glob(os.path.join(a.quick, "evals", "*.json"))):
        os.makedirs(os.path.join(OUT, "evals", "quickpath-fp16-check"), exist_ok=True)
        shutil.copyfile(f, os.path.join(OUT, "evals", "quickpath-fp16-check", os.path.basename(f)))
        written.append("evals/quickpath-fp16-check/" + os.path.basename(f))
    li = os.path.join(a.quick, "ldata_index.json")
    if os.path.exists(li):
        written.append(dump("lltCapstoneGPT2-quickpath-data-index.json",
                            {"runId": "lltCapstoneGPT2-quickpath-data-index", **json.load(open(li))}))

# ---- compute record ----------------------------------------------------------------------------------------------
fpt = 6 * main["config"]["params"] + 12 * 12 * 768 * 1024
br_tokens = sum((b["branchOf"]["maxSteps"] - b["branchOf"]["fromStep"]) * TPS for b in branches.values())
s1, s2 = main["segments"]
lost_sec = secs("mainStart", "shutdown1GPU") - s1["trainSec"]
stages = [
    ("Setup, data prep and calibration", 1, secs("boot1GPU", "mainStart")),
    ("Training, steps 1 to 1,500", 1, s1["trainSec"]),
    ("Training past the step 1,500 checkpoint, lost at the resize", 1, round(lost_sec, 1)),
    ("Restart on 4 GPUs (boot, compile)", 4, round(secs("boot4GPU", "mainEnd") - s2["trainSec"], 1)),
    ("Training, steps 1,501 to 19,073", 4, s2["trainSec"]),
    ("Five branch decays", 4, secs("mainEnd", "branchesEnd")),
    ("Evaluation failed on a missing package, VM idle", 4, secs("branchesEnd", "evalRelaunch1")),
    ("Second evaluation attempt and its fix", 4, secs("evalRelaunch1", "evalRelaunch2")),
]
if TIMELINE.get("evalsEnd"):
    stages.append(("Evaluations, and the learner quick-path check on 2 of the GPUs", 4, secs("evalRelaunch2", "evalsEnd")))
rows = [{"stage": n, "gpus": g, "wallSec": w, "gpuHours": round(g * w / 3600, 3)} for n, g, w in stages]
comp = {
    "runId": "lltCapstoneGPT2-ref-compute", "module": "lltCapstoneGPT2", "timelineUTC": TIMELINE,
    "flopsPerToken": fpt, "flopsFormula": "6N + 12*L*d*T, N = 123,551,232, L = 12, d = 768, T = 1,024",
    "mainTokens": m[-1]["tokens"], "mainFlops": fpt * m[-1]["tokens"],
    "branchTokens": br_tokens, "branchFlops": fpt * br_tokens,
    "peakFlopsPerGpu": PEAK, "segments": main["segments"],
    "stages": rows, "gpuHoursAttached": round(sum(r["gpuHours"] for r in rows), 3),
    "gpuHoursTraining": round(sum(r["gpuHours"] for r in rows if r["stage"].startswith("Training, ")), 3),
    "branchTrainSec": {k: v["branchTrainSec"] for k, v in branches.items()},
    "note": "GPU-hours count every GPU attached to the VM, busy or idle. The stages end when the evaluations end; "
            "time after that (copying results, the Hub upload) is not counted.",
}
written.append(dump("lltCapstoneGPT2-ref-compute.json", comp))
print("\n".join(written))
