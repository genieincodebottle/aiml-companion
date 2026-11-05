"""IsoFLOP fit of N_opt vs C, NumPy only (runs in the lab page under Pyodide 0.27).

fit_isoflop(runs, budgets) takes runs = [(N, [(tokens, val_loss), ...]), ...]. For each compute budget
C it reads every run's loss at D = C / (6N) (log-linear interpolation, no extrapolation), fits a
parabola in log10 N, and takes its minimum as N_opt(C). Then it fits log10 N_opt = a log10 C + b.
bootstrap() resamples runs with replacement and reports the 2.5 and 97.5 percentiles of a.

python scaling_fit.py synth   recovers a known exponent from the Chinchilla parametric law
python scaling_fit.py ours    runs the same code on our Run Library files
"""
import glob, json, os, sys
import numpy as np


def loss_at(points, D):
    d = np.log10([p[0] for p in points]); l = np.array([p[1] for p in points])
    x = np.log10(D)
    if x < d.min() or x > d.max():
        return None
    return float(np.interp(x, d, l))


def n_opt(runs, C):
    xs, ys = [], []
    for N, pts in runs:
        L = loss_at(pts, C / (6 * N))
        if L is not None:
            xs.append(np.log10(N)); ys.append(L)
    if len(set(xs)) < 3:
        return None
    a, b, _ = np.polyfit(xs, ys, 2)
    if a <= 0:
        return None
    x_min = -b / (2 * a)
    if x_min < min(xs) or x_min > max(xs):   # minimum outside the sizes we ran: not measured
        return None
    return 10 ** x_min


def fit_isoflop(runs, budgets):
    pairs = [(C, n_opt(runs, C)) for C in budgets]
    pairs = [(C, n) for C, n in pairs if n is not None]
    if len(pairs) < 3:
        return None, pairs
    a, b = np.polyfit(np.log10([p[0] for p in pairs]), np.log10([p[1] for p in pairs]), 1)
    return a, pairs


def bootstrap(runs, budgets, reps=500, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        a, _ = fit_isoflop([runs[i] for i in rng.integers(0, len(runs), len(runs))], budgets)
        if a is not None:
            out.append(a)
    if len(out) < reps // 2:
        return None, len(out)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))), len(out)


def synth(noise=0.0, seed=0):
    # Chinchilla approach 3 fit (Hoffmann et al. 2022): L = E + A / N^alpha + B / D^beta.
    E, A, B, al, be = 1.69, 406.4, 410.7, 0.34, 0.28
    rng = np.random.default_rng(seed)
    runs = []
    for N in np.logspace(7, 9, 9):
        D = np.logspace(8, 11, 31)
        runs.append((N, [(d, E + A / N ** al + B / d ** be + rng.normal(0, noise)) for d in D]))
    budgets = np.logspace(18, 20, 9)
    a, pairs = fit_isoflop(runs, budgets)
    ci, ok = bootstrap(runs, budgets)
    return {"trueExponent": round(be / (al + be), 4), "fitted": a and round(a, 4), "budgetsUsed": len(pairs),
            "ci95": ci and [round(c, 4) for c in ci], "bootstrapFitsOk": ok, "noise": noise}


def ours():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "..", "..", "runs", "lltTinyStories", "lltTinyStories-T2-*.json")))
    runs, used = [], []
    for f in files:
        r = json.load(open(f))
        if r.get("label") not in ("baseline", "ablation:size=3M", "ablation:size=30M") or "resume" in r["runId"]:
            continue
        pts = [(m["tokens"], m["valLoss"]) for m in r["metrics"] if m["step"] > 0 and m.get("valLoss") is not None]
        runs.append((r["params"], pts)); used.append(r["runId"])
    lo = max(min(6 * N * p[0][0] for N, p in runs), 1)
    hi = max(6 * N * p[-1][0] for N, p in runs)
    budgets = np.logspace(np.log10(lo), np.log10(hi), 25)
    a, pairs = fit_isoflop(runs, budgets)
    overlap = [C for C in budgets if sum(loss_at(p, C / (6 * N)) is not None for N, p in runs) >= 3 and len(set(N for N, p in runs if loss_at(p, C / (6 * N)) is not None)) >= 3]
    return {"runs": used, "budgetsTried": len(budgets), "budgetsWithThreeSizes": len(overlap),
            "overlapRange": [float("%.3g" % min(overlap)), float("%.3g" % max(overlap))] if overlap else None,
            "nOptFound": [[float("%.3g" % C), float("%.3g" % n)] for C, n in pairs], "exponent": a}


if __name__ == "__main__":
    if sys.argv[1] == "synth":
        print(json.dumps([synth(0.0), synth(0.01), synth(0.03)], indent=1))
    else:
        print(json.dumps(ours(), indent=1))
