import math
import numpy as np

def bt_loss(r_chosen, r_rejected):
    # Mean of -log sigmoid(r_chosen - r_rejected) over the pairs, stable for large margins of either sign.
    d = np.asarray(r_chosen, dtype=float) - np.asarray(r_rejected, dtype=float)
    return float(np.mean(np.logaddexp(0, -d)))

def kl_best_of_n(n):
    # The KL formula of Gao et al. 2023, in nats: an upper bound on the true KL of best-of-n (Beirami et al. 2024).
    return math.log(n) - (n - 1) / n

def best_of_n(select, value, n):
    # select, value: (P, N) arrays, N recorded samples per prompt. Return the expected value of the sample with the
    # highest select score among n samples drawn without replacement, averaged over the P prompts.
    P, N = select.shape
    weight = np.array([math.comb(i, n - 1) for i in range(N)], dtype=float) / math.comb(N, n)
    total = 0.0
    for p in range(P):
        order = np.argsort(select[p])
        total += float((weight * value[p][order]).sum())
    return total / P

# Checks. Run the cell; it prints All checks passed when all three functions are right.
assert abs(bt_loss([0.5], [0.0]) - 0.474077) < 1e-6, bt_loss([0.5], [0.0])
assert abs(bt_loss([0.0, 0.0], [0.0, 0.0]) - math.log(2)) < 1e-12, "equal rewards give ln 2"
assert abs(bt_loss([1000.0], [0.0])) < 1e-9 and abs(bt_loss([0.0], [1000.0]) - 1000.0) < 1e-9, "use a stable form"
assert abs(bt_loss([3.5, 1.0], [3.0, 2.0]) - bt_loss([0.5, -1.0], [0.0, 0.0])) < 1e-12, "only the margin matters"
assert kl_best_of_n(1) == 0.0
assert abs(kl_best_of_n(64) - 3.174508) < 1e-6, kl_best_of_n(64)
rng = np.random.default_rng(0)
sel = rng.normal(size=(5, 8))
val = sel + rng.normal(size=(5, 8))
assert abs(best_of_n(sel, val, 1) - val.mean()) < 1e-12, "n = 1 is the plain mean"
assert abs(best_of_n(sel, val, 8) - np.mean([val[p][np.argmax(sel[p])] for p in range(5)])) < 1e-12, "n = N is the argmax"
mc = np.mean([val[p][max(c, key=lambda j: sel[p][j])] for p in range(5) for c in
              [rng.choice(8, 3, replace=False) for _ in range(20000)]])
assert abs(best_of_n(sel, val, 3) - mc) < 0.02, (best_of_n(sel, val, 3), mc)
print("loss at margin 0.5", round(bt_loss([0.5], [0.0]), 4), "| KL at n = 64", round(kl_best_of_n(64), 4),
      "| best of 3", round(best_of_n(sel, val, 3), 4))
print("All checks passed")
