import numpy as np

def log_softmax(z):
    # Row-wise log of the softmax, stable for logits in the thousands.
    raise NotImplementedError

def divergence(s_logits, t_logits, beta):
    # Per-row divergence. beta 0: forward KL(teacher || student). beta 1: reverse KL(student || teacher).
    # 0 < beta < 1: beta KL(t || m) + (1 - beta) KL(s || m) with m = beta * teacher + (1 - beta) * student.
    raise NotImplementedError

def token_reward(s_logits, t_logits, token):
    # The on-policy signal for one sampled token: log p_teacher(token) - log p_student(token).
    raise NotImplementedError

# Checks. Run the cell; it prints All checks passed when all three functions are right.
s, t = np.array([0.0, 0.0]), np.array([np.log(3.0), 0.0])          # student 50/50, teacher 75/25
assert abs(divergence(s, t, 0) - 0.130812) < 1e-6, divergence(s, t, 0)
assert abs(divergence(s, t, 1) - 0.143841) < 1e-6, divergence(s, t, 1)
assert np.allclose(log_softmax([1000.0, 0.0]), [0.0, -1000.0]), "subtract the max before exp"
rng = np.random.default_rng(0)
a, b = rng.normal(size=(4, 50)), rng.normal(size=(4, 50))
for beta in (0, 0.5, 1):
    assert np.allclose(divergence(a, a, beta), 0, atol=1e-12), "a distribution has zero divergence from itself"
    assert np.allclose(divergence(a + 7.0, b, beta), divergence(a, b, beta)), "adding a constant to logits changes nothing"
assert np.allclose(divergence(a, b, 0.5), divergence(b, a, 0.5)), "JSD at beta 0.5 is symmetric"
assert (divergence(3 * a, -3 * a, 0.5) <= np.log(2) + 1e-12).all(), "JSD at beta 0.5 never exceeds ln 2"
ps = np.exp(log_softmax(a[0]))
draws = rng.choice(50, size=20000, p=ps)
mc = np.mean([token_reward(a[0], b[0], k) for k in draws])
assert abs(mc + divergence(a[0], b[0], 1)) < 0.05, (mc, divergence(a[0], b[0], 1))
print("forward", round(float(divergence(s, t, 0)), 6), "| reverse", round(float(divergence(s, t, 1)), 6),
      "| mean token reward over student samples", round(float(mc), 3), "| minus reverse KL", round(float(-divergence(a[0], b[0], 1)), 3))
print("All checks passed")