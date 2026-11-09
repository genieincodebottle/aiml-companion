"""Turn session records (llt_dpo_<kernel>.json from Kaggle or the GCP job) into Run Library files in runs/lltDPO/.

  python record_gpu.py <record.json> [<record.json> ...]
One run file per trained arm, one for DPOTrainer, one per session. Numbers are copied, not computed, except that the
training log gets the Run Library keys (step, trainLoss). A GCP session retrains SFT and the main DPO run; their files get
the suffix -gcp so the Kaggle core files are kept.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

HW = {"gpu": "Tesla T4", "count": 1, "vramGB": 15.6}
ARMS = ["dpo-b0.1-s0", "dpo-b0.01-s0", "dpo-b0.5-s0", "dpo-b0.1-s1", "ipo-s0", "simpo-s0", "orpo-s0", "kto-s0",
        "onpolicy-gold-s0", "onpolicy-proxy-s0", "offline-gold-s0"]
# The Kaggle core session predates the per-step SFT loss; these are its printed means of the last 75 steps
CORE_SFT_PRINTED = [(1, 1.1247), (75, 1.3158), (150, 1.2554), (225, 1.223), (300, 1.2471), (375, 1.2159)]


def steps_log(log):
    return [{"step": i + 1, "trainLoss": r["loss"], "chosenLogRatio": r["chosen"], "rejectedLogRatio": r["rejected"]}
            for i, r in enumerate(log)]


def main():
    for path in sys.argv[1:]:
        rec = json.load(open(path, encoding="utf-8"))
        k, runs = rec["kernel"], rec["runs"]
        gcp = rec.get("host") == "gcp"
        env = dict(rec["env"], image=("Google Cloud T4 VM llt-c2, " if gcp else "Kaggle GPU image, ") + rec["env"]["torch"])
        src = {"kernel": ("gcp:llt-dpo-" if gcp else "llt-dpo-") + k, "file": Path(path).name, "restarts": rec.get("restarts", 0)}
        sfx = ("-gcp" if k == "exp-gcp" else "-" + k) if gcp else ""   # one file per session for runs every session repeats
        if "sft" in runs:
            s = runs["sft"]
            if "loss" in s:
                metrics, note = [{"step": i + 1, "trainLoss": v} for i, v in enumerate(s["loss"])], None
            else:
                metrics = [{"step": st, "trainLossMeanLast75": v} for st, v in CORE_SFT_PRINTED]
                note = "per-step loss not kept in this session; rows are the cell's printed means of the last 75 steps"
            write_run("lltDPO-T2-sft-s0" + sfx, "T2", "baseline", {"model": "Qwen/Qwen3-0.6B-Base", "method": "LoRA r16 all-linear, merged",
                      "pairs": 3000, "lr": 2e-4, "batch": 8, "maxLen": 512, **src}, 0, metrics,
                      evals={kk: v for kk, v in s.items() if kk != "loss"}, hardware=HW, env=env, wall_clock_sec=s["sec"],
                      notes=note)
        for arm in ARMS:
            if arm not in runs:
                continue
            r = runs[arm]
            evals = {key: r.get(key) for key in ("final", "gold", "vsSFT", "vsSFT2", "tokens", "ended", "perPair") if key in r}
            if arm == "dpo-b0.1-s0":
                if "judge" in runs:
                    evals["vsSFT"] = runs["judge"]
                if "judge2" in runs:
                    evals["vsSFT2"] = runs["judge2"]
                if "finalHeld" in rec:
                    evals["perPair"] = {"chosen": [round(x, 2) for x in rec["finalHeld"]["dc"]],
                                        "rejected": [round(x, 2) for x in rec["finalHeld"]["dr"]]}
            label = "baseline" if arm == "dpo-b0.1-s0" else "ablation:" + arm
            write_run("lltDPO-T2-" + arm + (sfx if arm == "dpo-b0.1-s0" else ""), "T2", label,
                      {"kind": r["kind"], "beta": r["beta"], "lr": r["lr"], "pairs": r["pairs"], "epochs": r.get("epochs", 1),
                       "steps": r["steps"], **src}, r["seed"], steps_log(r["log"]), evals={"held": r["held"], **evals},
                      hardware=HW, env=env, wall_clock_sec=r["sec"],
                      notes={k2: r.get(k2) for k2 in ("trainSec", "peakGB", "trainPeakGB", "evalPeakGB") if k2 in r})
        if "trl" in runs:
            t = runs["trl"]
            write_run("lltDPO-T2-trl" + sfx, "T2", "baseline", {"trainer": "DPOTrainer", "steps": 40, "loggingSteps": 10, **src}, 0,
                      [dict(h, step=10 * (i + 1)) for i, h in enumerate(t["log"]) if "loss" in h],
                      evals={"sPerStep": t["sPerStep"], "peakGB": t["peakGB"]}, hardware=HW, env=env, wall_clock_sec=t["sec"])
        session = {key: v for key, v in runs.items() if key not in ARMS and key not in ("sft", "trl")}
        write_run("lltDPO-T2-notebook-" + k, "T2", "baseline", {"cellTimesSec": rec["cellTimes"], "cellPeakGB": rec.get("cellPeakGB"), **src},
                  0, [], evals=session,
                  samples=[{"gens": rec.get("gens"), "gold": rec.get("gold"), "judge2": rec.get("judge2")}],
                  hardware=HW, env=env, wall_clock_sec=rec["totalSec"])


if __name__ == "__main__":
    main()
