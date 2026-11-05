import numpy as np

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def grad_norm(p):
    # Size of the softmax gradient, the Frobenius norm of diag(p) - p p^T, per row
    sq = (p ** 2).sum(-1)
    return np.sqrt(np.maximum(sq - 2 * (p ** 3).sum(-1) + sq ** 2, 0))

rng = np.random.default_rng(0)
n, T = 4_000, 64            # n query-key pairs for the variance, T keys per query
print("     d   var(q.k)   var / d")
for d in [16, 64, 256, 1024]:
    q, k = rng.standard_normal((n, d)), rng.standard_normal((n, d))
    dots = (q * k).sum(axis=1)                  # n dot products of unit-variance vectors
    print(f"{d:6d} {dots.var():10.1f} {dots.var() / d:9.3f}")

print()
print("     d   top weight        rows above 0.99   median gradient")
print("         raw    scaled     raw    scaled     raw      scaled")
for d in [16, 64, 256, 1024]:
    Q, K = rng.standard_normal((500, d)), rng.standard_normal((T, d))
    raw = Q @ K.T                               # 500 queries, each scored against T keys
    pr, ps = softmax(raw), softmax(raw / np.sqrt(d))
    print(f"{d:6d}   {pr.max(-1).mean():.3f}  {ps.max(-1).mean():.3f}     "
          f"{(pr.max(-1) > 0.99).mean():.3f}  {(ps.max(-1) > 0.99).mean():.3f}     "
          f"{np.median(grad_norm(pr)):.5f}  {np.median(grad_norm(ps)):.3f}")
