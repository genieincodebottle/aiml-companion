# calib_summary.py - read the calibration runs and project the full reference run's time and cost.
# Usage: python calib_summary.py <workdir> <price_per_hour_usd>
import glob, json, os, statistics, sys

W, price = sys.argv[1], float(sys.argv[2])
MAIN_STEPS, BRANCH_STEPS, TOKENS_PER_STEP = 19_073, 48 + 95 + 240 + 477 + 477, 524_288
out = {"pricePerHourUSD": price, "runs": {}}
for f in sorted(glob.glob(os.path.join(W, "cal*", "cal-micro*.json"))):
    d = json.load(open(f))
    ms = [m for m in d["metrics"] if m["step"] > 10]                 # skip compile and warm-up steps
    tps = statistics.median(m["tokPerSec"] for m in ms)
    out["runs"][d["runId"]] = {"micro": d["config"]["micro"], "medianTokPerSec": tps,
                               "medianMfu": statistics.median(m["mfu"] for m in ms),
                               "peakMemGB": max(m["memGB"] for m in d["metrics"]), "gpu": d["hardware"]["gpu"]}
best = max(out["runs"].values(), key=lambda r: r["medianTokPerSec"])
train_h = (MAIN_STEPS + BRANCH_STEPS) * TOKENS_PER_STEP / best["medianTokPerSec"] / 3600
prep = os.path.join(W, "data_cal", "index.json")
if os.path.exists(prep):
    ix = json.load(open(prep))
    out["prepOneFile"] = {"sec": ix["sec"], "tokens": ix["trainTokens"] + ix["val"]["tokens"], "perFile": ix["perFile"]}
out["best"] = best
out["projectedTrainHours"] = round(train_h, 2)
out["projectedTrainCostUSD"] = round(train_h * price, 2)
print(json.dumps(out, indent=1))
with open(os.path.join(W, "run", "calibration.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"\nPick MICRO={best['micro']}. Training + branch decays: about {train_h:.1f} h, about ${train_h * price:.0f} "
      f"at ${price}/h. Add data prep (see prepOneFile x 14 / workers) and about 1 h of evals.")
