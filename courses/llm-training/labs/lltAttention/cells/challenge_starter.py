import numpy as np

def split_heads(a, h):
    # TODO: (B, T, C) -> (B, h, T, C // h). Head i must hold columns i*d to (i+1)*d - 1.
    return None

def merge_heads(a):
    # TODO: (B, h, T, d) -> (B, T, h * d), the exact inverse of split_heads
    return None

def causal_mask(T):
    # TODO: a (T, T) boolean array, True where key j is in the future of query i (j > i)
    return None

def stable_softmax(s):
    # TODO: softmax over the last axis that works for scores like 1000 and for -inf
    return None

def attention(x, Wqkv, Wo, h):
    q, k, v = (split_heads(a, h) for a in np.split(x @ Wqkv, 3, axis=-1))
    T, d = q.shape[-2:]
    scores = np.where(causal_mask(T), -np.inf, q @ k.swapaxes(-1, -2) / np.sqrt(d))
    return merge_heads(stable_softmax(scores) @ v) @ Wo

# Checks. Run the cell. It prints All checks passed when all four functions are right.
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
