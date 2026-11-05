import numpy as np

def split_heads(a, h):
    # (B, T, C) -> (B, T, h, d) -> (B, h, T, d). The transpose keeps each token's row intact.
    B, T, C = a.shape
    return a.reshape(B, T, h, C // h).transpose(0, 2, 1, 3)

def merge_heads(a):
    # Undo the transpose first, then join the heads back into C columns
    B, h, T, d = a.shape
    return a.transpose(0, 2, 1, 3).reshape(B, T, h * d)

def causal_mask(T):
    # Strictly above the diagonal: key j > query i
    return np.triu(np.ones((T, T), dtype=bool), k=1)

def stable_softmax(s):
    # Subtracting the row max changes nothing mathematically and keeps exp() at or below 1
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def attention(x, Wqkv, Wo, h):
    q, k, v = (split_heads(a, h) for a in np.split(x @ Wqkv, 3, axis=-1))
    T, d = q.shape[-2:]
    scores = np.where(causal_mask(T), -np.inf, q @ k.swapaxes(-1, -2) / np.sqrt(d))
    return merge_heads(stable_softmax(scores) @ v) @ Wo

rng = np.random.default_rng(0)
B, T, C, h = 2, 6, 16, 4
x = rng.standard_normal((B, T, C))
assert split_heads(x, h) is not None, "split_heads returns None"
assert split_heads(x, h).shape == (2, 4, 6, 4), "split_heads should give (B, h, T, d)"
assert np.array_equal(split_heads(x, h)[:, 1], x[:, :, 4:8]), "head 1 should hold columns 4 to 7"
assert np.array_equal(merge_heads(split_heads(x, h)), x), "merge_heads(split_heads(x)) should give x back"
assert causal_mask(3) is not None, "causal_mask returns None"
assert causal_mask(3).tolist() == [[False, True, True], [False, False, True], [False, False, False]], "mask is wrong"
p = stable_softmax(np.array([[1000.0, 1001.0, 1002.0], [0.0, -np.inf, 1.0]]))
assert p is not None and not np.isnan(p).any(), "softmax gives NaN, subtract the row max first"
assert np.allclose(p[0], [0.09003057, 0.24472847, 0.66524096]), "softmax values are wrong"
assert p[1, 1] == 0.0 and np.allclose(p.sum(-1), 1.0), "-inf should get weight 0 and rows should sum to 1"
Wqkv, Wo = rng.standard_normal((C, 3 * C)) / 4, rng.standard_normal((C, C)) / 4
y = attention(x, Wqkv, Wo, h)
for t in range(T - 1):
    x2 = x.copy()
    x2[:, t + 1] += 1.0
    moved = np.abs(attention(x2, Wqkv, Wo, h)[:, : t + 1] - y[:, : t + 1]).max()
    assert moved == 0.0, f"changing token {t + 1} moved the output at positions <= {t}"
print("y", y.shape, "| leakage test passed for t = 0 to", T - 2)
print("All checks passed")
