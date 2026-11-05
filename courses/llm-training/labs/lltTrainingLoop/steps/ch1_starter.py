import numpy as np


def adamw_step(p, g, m, v, t, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.1):
    # TODO: one AdamW update at step t (t starts at 1). Return the new p, m and v.
    # 1. Shrink p by the decoupled weight decay, p * (1 - lr * wd).
    # 2. Update m and v, the moving averages of g and g squared.
    # 3. Correct both for their zero start, then take the Adam step.
    return p, m, v


p = np.array([0.5, -1.0, 2.0, 0.0])
grads = [[0.1, -0.2, 0.3, 0.4], [0.05, -0.1, -0.2, 0.3], [-0.3, 0.2, 0.1, 0.0]]
m, v = np.zeros(4), np.zeros(4)
for t, g in enumerate(grads, start=1):
    p, m, v = adamw_step(p, np.array(g), m, v, t)
    print("step", t, np.round(p, 8))

# Checks. torch.optim.AdamW(lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1) gave these values.
expected = np.array([0.32957609, -0.76892492, 1.80042958, -0.27123826])
assert not np.allclose(p, [0.5, -1.0, 2.0, 0.0]), "p did not change. Fill in adamw_step."
assert np.allclose(p, expected, atol=1e-7), "Step 3 does not match torch. Check the decay order and the bias correction."
print("All checks passed")
