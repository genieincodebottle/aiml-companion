import hashlib
import numpy as np

P = 2**31 - 1
rng = np.random.default_rng(1)
K, BANDS, ROWS = 112, 14, 8
A = rng.integers(1, P, size=K, dtype=np.uint64)
B = rng.integers(0, P, size=K, dtype=np.uint64)

def shingles(text, n=5):
    w = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    grams = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return np.array([int(hashlib.sha1(g.encode()).hexdigest()[:8], 16) % P for g in grams], dtype=np.uint64)

def minhash(sh):
    return ((sh[:, None] * A[None, :] + B[None, :]) % P).min(axis=0)

def jaccard(s1, s2):
    s1, s2 = set(s1.tolist()), set(s2.tolist())
    return len(s1 & s2) / len(s1 | s2)

base = ("The library opens at nine every morning and closes at six in the evening. Children come "
        "after school to read stories, and older readers sit near the back with the newspapers. "
        "On rainy days the reading room is full, and the staff put out extra chairs. The building "
        "was restored last year with a new roof and better lighting over the long tables. ")
others = {
    "same page, new footer": base + "Copyright 2024. All rights reserved.",
    "same page, one sentence changed": base.replace("extra chairs", "folding chairs and tables"),
    "different page": "Preheat the oven to 180 degrees. Mix two cups of flour with one egg, a cup of milk "
                      "and a pinch of salt. Pour the batter into a greased tin and bake it for forty minutes.",
}
s0 = shingles(base + "Contact the front desk for opening hours.")
m0 = minhash(s0)
for name, text in others.items():
    s1 = shingles(text)
    m1 = minhash(s1)
    agree = m0 == m1
    bands_hit = sum(agree[b * ROWS:(b + 1) * ROWS].all() for b in range(BANDS))
    print(f"{name:32} Jaccard {jaccard(s0, s1):.3f}  MinHash estimate {agree.mean():.3f}  bands matched {bands_hit}")

print("P(candidate) = 1 - (1 - J**8)**14")
for J in [0.3, 0.5, 0.7, 0.8, 0.9]:
    print(f"  J = {J:.1f}   {1 - (1 - J**ROWS)**BANDS:.4f}")
