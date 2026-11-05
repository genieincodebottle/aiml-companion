import numpy as np

def rope(x, pos, base=10000.0):
    # Rotate each pair (x[2i], x[2i+1]) by the angle pos * base**(-2i/d)
    d = x.shape[-1]
    ang = pos * base ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out

rng = np.random.default_rng(0)
q, k = rng.normal(size=64), rng.normal(size=64)

# A query at position m and a key at position n, always 3 apart
for m in [3, 10, 100, 1000]:
    print(f"RoPE         m = {m:>4}  n = {m - 3:>3}  score {rope(q, m) @ rope(k, m - 3):.6f}")
scores = [rope(q, m) @ rope(k, m - 3) for m in range(3, 2000)]
print(f"RoPE, 1,997 pairs with gap 3, max - min = {max(scores) - min(scores):.1e}")
print(f"RoPE, gap 5 instead   score {rope(q, 5) @ rope(k, 0):.6f}")
print(f"length of q {np.linalg.norm(q):.6f}, after rotating to position 1000 {np.linalg.norm(rope(q, 1000)):.6f}")

# The GPT-2 way: add a position vector to q and k instead of rotating them
P = rng.normal(size=(2000, 64))
for m in [3, 10, 100, 1000]:
    print(f"added vector m = {m:>4}  n = {m - 3:>3}  score {(q + P[m]) @ (k + P[m - 3]):.6f}")
