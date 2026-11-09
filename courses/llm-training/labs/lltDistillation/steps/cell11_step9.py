def paired(a, b, reps=10000):
    # accuracy of arm a minus arm b on the same test questions, with a 95% paired bootstrap interval
    x, y = np.array(R[a]["correct"], float), np.array(R[b]["correct"], float)
    idx = np.random.default_rng(0).integers(0, len(x), (reps, len(x)))
    d = (x[idx] - y[idx]).mean(1)
    return round(float((x - y).mean()), 3), round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)

base, seqkd = "student, before training", "SFT on teacher answers"
print("teacher accuracy", R["teacher"]["acc"], "| student before training", R[base]["acc"])
for name, r in R.items():
    if name in ("teacher", base):
        continue
    line = [name, "| accuracy", r["acc"], "| vs before", paired(name, base)]
    if name != seqkd:
        line += ["| vs SFT on teacher answers", paired(name, seqkd)]
    if "trainSec" in r:
        line += ["| training s", r["trainSec"], "of it sampling", r["sampleSec"]]
    print(*line)