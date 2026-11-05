import numpy as np

def kl_per_row(p, q):
    # p, q: arrays of shape (T, V), each row a probability distribution.
    # Return KL(p || q) for each row, in nats. A zero in p adds nothing (0 log 0 = 0).
    safe_p = np.where(p > 0, p, 1.0)
    return np.where(p > 0, p * np.log(safe_p / q), 0.0).sum(axis=1)

def base_rank(q, tokens):
    # q: (T, V) base-model distributions. tokens: the token id the instruct model chose at each row.
    # Rank 1 means the base model also put that token first. Ties count in the token's favour.
    chosen = q[np.arange(len(tokens)), tokens]
    return (q > chosen[:, None]).sum(axis=1) + 1

def shift_label(rank):
    # URIAL's three groups: rank 1 unshifted, rank 2 or 3 marginal, above 3 shifted.
    return np.where(rank == 1, "unshifted", np.where(rank <= 3, "marginal", "shifted"))

# Checks. Run the cell; it prints All checks passed when all three functions are right.
p = np.array([[0.90, 0.05, 0.03, 0.02, 0.00],
              [0.02, 0.92, 0.03, 0.02, 0.01],
              [0.01, 0.02, 0.95, 0.01, 0.01],
              [0.01, 0.01, 0.02, 0.06, 0.90]])
q = np.array([[0.10, 0.60, 0.20, 0.05, 0.05],
              [0.05, 0.85, 0.05, 0.03, 0.02],
              [0.02, 0.03, 0.90, 0.03, 0.02],
              [0.15, 0.30, 0.30, 0.20, 0.05]])
kl = kl_per_row(p, q)
assert kl.shape == (4,), "return one KL value per row"
assert np.all(np.isfinite(kl)), "a zero in p must add 0, not nan"
assert np.allclose(kl, [1.7780, 0.0241, 0.0184, 2.4138], atol=1e-4), np.round(kl, 4)
assert np.allclose(kl_per_row(q, q), 0.0), "KL of a distribution with itself is 0"
assert kl_per_row(q, p[[1, 1, 2, 3]]).min() >= 0, "KL is never negative"
tokens = p.argmax(axis=1)
rank = base_rank(q, tokens)
assert rank.tolist() == [3, 1, 1, 5], rank.tolist()
assert base_rank(np.array([[0.5, 0.5]]), np.array([1])).tolist() == [1], "a tie with the top token is rank 1"
assert shift_label(rank).tolist() == ["marginal", "unshifted", "unshifted", "shifted"], shift_label(rank).tolist()
share = kl / kl.sum()
assert round(float(share[[0, 3]].sum()), 3) == 0.990, round(float(share[[0, 3]].sum()), 3)
print("KL per row", np.round(kl, 4), "| base ranks", rank.tolist())
print("Rows 1 and 4 hold", f"{share[[0, 3]].sum():.1%}", "of the total KL")
print("All checks passed")
