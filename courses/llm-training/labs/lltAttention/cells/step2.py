import numpy as np

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def split_heads(a, h):          # (B, T, C) -> (B, h, T, d), d = C // h
    B, T, C = a.shape
    return a.reshape(B, T, h, C // h).transpose(0, 2, 1, 3)

def merge_heads(a):             # (B, h, T, d) -> (B, T, h * d)
    B, h, T, d = a.shape
    return a.transpose(0, 2, 1, 3).reshape(B, T, h * d)

def attend(q, k, v):            # (B, h, T, d) each -> (B, h, T, d)
    T, d = q.shape[-2:]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)               # (B, h, T, T)
    scores = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, scores)
    return softmax(scores) @ v

def mha(x, Wqkv, Wo, h, split=split_heads):
    q, k, v = np.split(x @ Wqkv, 3, axis=-1)                    # each (B, T, C)
    return merge_heads(attend(split(q, h), split(k, h), split(v, h))) @ Wo

rng = np.random.default_rng(0)
B, T, C, h = 2, 6, 16, 4
d = C // h
x = rng.standard_normal((B, T, C))
Wqkv = rng.standard_normal((C, 3 * C)) / np.sqrt(C)
Wo = rng.standard_normal((C, C)) / np.sqrt(C)
y = mha(x, Wqkv, Wo, h)

# Reference: each head on its own slice of d columns, one head at a time
q, k, v = np.split(x @ Wqkv, 3, axis=-1)
heads = [attend(q[:, None, :, i * d:(i + 1) * d], k[:, None, :, i * d:(i + 1) * d],
                v[:, None, :, i * d:(i + 1) * d])[:, 0] for i in range(h)]
ref = np.concatenate(heads, axis=-1) @ Wo
print("x", x.shape, "| one head of q", split_heads(q, h).shape, "| y", y.shape)
print("max abs difference from the per-head loop", np.abs(y - ref).max())

# A common bug: reshape straight to (B, h, T, d) with no transpose
wrong = lambda a, h: a.reshape(a.shape[0], h, a.shape[1], -1)
print("same shape?", mha(x, Wqkv, Wo, h, wrong).shape == y.shape,
      "| max abs difference", round(float(np.abs(mha(x, Wqkv, Wo, h, wrong) - ref).max()), 3))
