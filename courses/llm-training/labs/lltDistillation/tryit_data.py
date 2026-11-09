"""Reference data for the lltDistillation Try it widget (1D distribution fitter with a KL-direction toggle).

Same grid and teacher as page cell 1. The widget recomputes everything in the browser; this file gives the values its
tests must reproduce. Writes llt6/lltDistillation.tryit_data.json (path given as argv[1]).
"""
import json, sys
import numpy as np

x = np.linspace(-6, 6, 1201)
dx = x[1] - x[0]


def normal(mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def kl(a, b):
    return float(np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx)


mus, sds = np.arange(-4, 4.001, 0.05), np.arange(0.2, 4.001, 0.05)
cases = []
for w in (0.5, 0.6, 0.8):
    for gap in (1.0, 2.0, 3.0):
        teacher = w * normal(-gap, 0.6) + (1 - w) * normal(gap, 0.6)
        F = np.array([[kl(teacher, normal(m, s)) for s in sds] for m in mus])
        Rv = np.array([[kl(normal(m, s), teacher) for s in sds] for m in mus])
        fi, ri = np.unravel_index(F.argmin(), F.shape), np.unravel_index(Rv.argmin(), Rv.shape)
        best_sd_per_mu_rev = [round(float(Rv[i].min()), 4) for i in range(len(mus))]   # reverse-KL landscape over the mean
        valley = np.abs(x) < 0.5
        fs, rs = normal(mus[fi[0]], sds[fi[1]]), normal(mus[ri[0]], sds[ri[1]])
        cases.append({"leftWeight": w, "modeAt": gap, "modeSd": 0.6,
                      "forwardFit": {"mean": round(float(mus[fi[0]]), 2), "sd": round(float(sds[fi[1]]), 2),
                                     "forwardKL": round(float(F.min()), 4), "reverseKL": round(kl(fs, teacher), 4),
                                     "valleyMass": round(float(fs[valley].sum() * dx), 4)},
                      "reverseFit": {"mean": round(float(mus[ri[0]]), 2), "sd": round(float(sds[ri[1]]), 2),
                                     "reverseKL": round(float(Rv.min()), 4), "forwardKL": round(kl(teacher, rs), 4),
                                     "valleyMass": round(float(rs[valley].sum() * dx), 4)},
                      "teacherValleyMass": round(float(teacher[valley].sum() * dx), 4),
                      "reverseKLBestOverSdByMean": best_sd_per_mu_rev})
json.dump({"grid": {"x": [-6, 6, 1201], "mean": [-4, 4, 0.05], "sd": [0.2, 4, 0.05]}, "means": [round(float(m), 2) for m in mus],
           "cases": cases, "source": "labs/lltDistillation/tryit_data.py, same functions as page cell 1"},
          open(sys.argv[1], "w"), indent=1)
for c in cases:
    print(c["leftWeight"], c["modeAt"], c["forwardFit"], c["reverseFit"])