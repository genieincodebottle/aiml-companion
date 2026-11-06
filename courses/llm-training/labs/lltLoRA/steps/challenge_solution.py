import numpy as np

def lora_forward(x, W, A, B, alpha, rslora=False):
    # x: (n, d_in). W: (d_out, d_in), frozen. A: (r, d_in). B: (d_out, r).
    # Return x W^T plus the scaled LoRA path. The scale is alpha / r, or alpha / sqrt(r) with rslora.
    r = A.shape[0]
    scale = alpha / np.sqrt(r) if rslora else alpha / r
    return x @ W.T + scale * (x @ A.T @ B.T)

def merge(W, A, B, alpha, rslora=False):
    # Return a new weight matrix that gives the same output as lora_forward with no extra path.
    r = A.shape[0]
    scale = alpha / np.sqrt(r) if rslora else alpha / r
    return W + scale * (B @ A)

def energy_top_r(delta, r):
    # Share of the squared Frobenius norm of delta held by its top r singular values (between 0 and 1).
    s = np.linalg.svd(delta, compute_uv=False)
    return float((s[:r] ** 2).sum() / (s ** 2).sum())

# Checks. Run the cell; it prints All checks passed when all three functions are right.
rng = np.random.default_rng(0)
d_in, d_out, r = 64, 96, 4
x = rng.standard_normal((5, d_in))
W = rng.standard_normal((d_out, d_in)) / 8
A = rng.standard_normal((r, d_in)) / 8
B0 = np.zeros((d_out, r))
assert np.allclose(lora_forward(x, W, A, B0, alpha=8), x @ W.T), "with B = 0 the output must equal the base layer"
B = rng.standard_normal((d_out, r)) / 8
y = lora_forward(x, W, A, B, alpha=8)
assert np.allclose(y - x @ W.T, 2.0 * (x @ A.T @ B.T)), "alpha 8 at r 4 gives a scale of 2.0"
y_rs = lora_forward(x, W, A, B, alpha=8, rslora=True)
assert np.allclose(y_rs - x @ W.T, 4.0 * (x @ A.T @ B.T)), "rsLoRA scale is alpha / sqrt(r) = 4.0"
Wm = merge(W, A, B, alpha=8)
assert Wm.shape == W.shape and not np.shares_memory(Wm, W), "return a new (d_out, d_in) matrix"
assert np.allclose(x @ Wm.T, y), "the merged layer must match the adapter output"
assert np.linalg.matrix_rank(Wm - W) == r, "a merged LoRA update has rank r"
assert abs(energy_top_r(Wm - W, r) - 1.0) < 1e-9, "a rank-r update keeps all its energy in r values"
D = np.diag([4.0, 2.0, 1.0, 1.0])
assert abs(energy_top_r(D, 1) - 16 / 22) < 1e-9, energy_top_r(D, 1)
assert abs(energy_top_r(D, 2) - 20 / 22) < 1e-9, energy_top_r(D, 2)
full = rng.standard_normal((d_out, d_in))
print("random full-rank matrix, energy in the top 4 of 64 values:", round(energy_top_r(full, 4), 3))
print("trainable values per layer:", r * (d_in + d_out), "of", d_in * d_out)
print("All checks passed")
