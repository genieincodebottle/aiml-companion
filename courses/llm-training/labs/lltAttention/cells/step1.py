import numpy as np

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)   # the largest score in each row becomes 0
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def causal_attention(x, Wq, Wk, Wv):
    q, k, v = x @ Wq, x @ Wk, x @ Wv                     # each (T, d)
    T, d = q.shape
    scores = q @ k.T / np.sqrt(d)                        # (T, T), row i = query i
    future = np.triu(np.ones((T, T), dtype=bool), k=1)   # True where key j > query i
    scores = np.where(future, -np.inf, scores)           # mask BEFORE the softmax
    w = softmax(scores)                                  # (T, T), each row sums to 1
    return w @ v, w                                      # (T, d) and (T, T)

rng = np.random.default_rng(0)
T, C, d = 6, 16, 8          # tokens, model width, head width
x = rng.standard_normal((T, C))
Wq, Wk, Wv = (rng.standard_normal((C, d)) / np.sqrt(C) for _ in range(3))

y, w = causal_attention(x, Wq, Wk, Wv)
print("x", x.shape, "| q, k, v", (x @ Wq).shape, "| weights", w.shape, "| y", y.shape)
print("row sums", np.round(w.sum(axis=1), 6))
print("weight on future tokens", w[np.triu(np.ones((T, T), dtype=bool), k=1)].sum())
print(np.round(w, 2))
