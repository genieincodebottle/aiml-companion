import numpy as np

words = ["mat", "sofa", "floor", "bed", "roof", "moon", "table", "piano"]
logits = np.array([3.2, 2.5, 2.1, 1.6, 0.4, -0.8, 1.1, -1.5])

def softmax(z):
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()

def temperature(z, t):
    return z / t

def top_k(z, k):
    kth = np.sort(z)[-k]
    return np.where(z >= kth, z, -np.inf)

def top_p(z, p):
    order = np.argsort(-z)
    probs = softmax(z)[order]
    before = np.cumsum(probs) - probs        # mass of the tokens ranked above each one
    keep = np.zeros(len(z), dtype=bool)
    keep[order[before < p]] = True
    return np.where(keep, z, -np.inf)

def show(name, z):
    p = softmax(z)
    kept = [f"{w} {q:.3f}" for w, q in sorted(zip(words, p), key=lambda x: -x[1]) if q > 0]
    print(f"{name:14} kept {len(kept)}  " + ", ".join(kept))

show("T = 1.0", logits)
show("T = 0.5", temperature(logits, 0.5))
show("T = 2.0", temperature(logits, 2.0))
show("top-k, k = 3", top_k(logits, 3))
show("top-p, p = 0.9", top_p(logits, 0.9))

rng = np.random.default_rng(0)
draws = rng.choice(len(words), size=1000, p=softmax(top_p(logits, 0.9)))
print("1,000 draws with top-p 0.9:", {words[i]: int(c) for i, c in zip(*np.unique(draws, return_counts=True))})
