import numpy as np


def adamw_step(p, g, m, v, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.1):
    p = p * (1 - lr * wd)                  # decoupled weight decay, applied to the weights directly
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)              # m and v start at zero, so early averages are too small
    v_hat = v / (1 - b2 ** t)
    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p, m, v


p = np.array([0.5, -1.0, 2.0, 0.0])
grads = [[0.1, -0.2, 0.3, 0.4], [0.05, -0.1, -0.2, 0.3], [-0.3, 0.2, 0.1, 0.0]]
m, v = np.zeros(4), np.zeros(4)
for t, g in enumerate(grads, start=1):
    p, m, v = adamw_step(p, np.array(g), m, v, t)
    print("step", t, np.round(p, 8))

expected = np.array([0.32957609, -0.76892492, 1.80042958, -0.27123826])
assert not np.allclose(p, [0.5, -1.0, 2.0, 0.0]), "p did not change. Fill in adamw_step."
assert np.allclose(p, expected, atol=1e-7), "Step 3 does not match torch. Check the decay order and the bias correction."
print("All checks passed")
