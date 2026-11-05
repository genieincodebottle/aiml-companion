"""Expected loss of random logits against ln(V) + s^2/2 (lltIntro step 2 and its derivation panel).

For each logit spread s, draws 64 rows of V normal logits, averages the loss over every possible
target, and repeats with 5 seeds. Writes runs/lltIntro/lltIntro-t0-logit-spread.json.
Usage: python simulate_logit_spread.py
"""
import numpy as np

from runlog import write_run


def expected_loss(V, s, seed, rows=64):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, s, size=(rows, V))
    m = z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z - m).sum(axis=1)) + m[:, 0]
    return float((lse - z.mean(axis=1)).mean())


cases = [(49_152, s) for s in (0.0, 0.16, 1.0, 2.0, 3.0, 4.0, 5.0, 24.0)] + [(63, 0.16), (63, 8.0), (32_000, 2.0132)]
metrics = []
for V, s in cases:
    vals = [expected_loss(V, s, seed) for seed in range(5)]
    row = {"V": V, "spread": s, "formula": round(float(np.log(V) + s * s / 2), 2),
           "simulatedBySeed": [round(v, 2) for v in vals], "simulatedMean": round(float(np.mean(vals)), 2),
           "simulatedMin": round(min(vals), 2), "simulatedMax": round(max(vals), 2)}
    metrics.append(row)
    print(row)
write_run("lltIntro-t0-logit-spread", "T0", "derivation-check", {"rows": 64, "seeds": [0, 1, 2, 3, 4],
          "method": "normal logits, loss averaged over every possible target"}, 0, metrics=metrics)
