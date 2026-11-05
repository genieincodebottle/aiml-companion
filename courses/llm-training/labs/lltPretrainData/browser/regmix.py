import numpy as np

# Our mixture runs: token weights for web, Wikipedia, stories, and the measured bits per byte
# on each held-out set (runs lltPretrainData-T2-mix-*, 300 steps of 64 x 512 tokens, 11M parameters).
RUNS = [
    ([0.3333, 0.3333, 0.3333], [2.0557, 1.9416, 1.1410]),  # equal by tokens
    ([0.1345, 0.8288, 0.0367], [2.0825, 1.8012, 1.4800]),  # equal by documents
    ([0.6, 0.2, 0.2], [2.0169, 1.9794, 1.2241]),           # web 60%
    ([0.2, 0.6, 0.2], [2.0550, 1.8344, 1.2113]),           # Wikipedia 60%
    ([0.2, 0.2, 0.6], [2.1246, 2.0278, 1.0603]),           # stories 60%
]
W = np.array([r[0] for r in RUNS])          # [n_runs, 3] mixture weights by tokens
Y = np.array([r[1] for r in RUNS])          # [n_runs, 3] bits per byte on web, wiki, stories

def fit_ridge(W, y, lam=1e-3):
    X = np.hstack([W, np.ones((len(W), 1))])
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def predict(coef, w):
    return np.append(w, 1.0) @ coef

names = ["web", "wiki", "stories"]
for d in range(3):
    errs = []
    for k in range(len(W)):             # leave one run out, fit on the rest, predict it
        rest = [i for i in range(len(W)) if i != k]
        errs.append(predict(fit_ridge(W[rest], Y[rest, d]), W[k]) - Y[k, d])
    print(f"{names[d]:8} leave-one-out error, mean absolute {np.mean(np.abs(errs)):.3f} bits per byte")
coef = [fit_ridge(W, Y[:, d]) for d in range(3)]
w_new = np.array([0.5, 0.3, 0.2])
print("prediction for web 0.5, wiki 0.3, stories 0.2:", [round(float(predict(c, w_new)), 3) for c in coef])
