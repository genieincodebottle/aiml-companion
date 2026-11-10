import numpy as np

def ppo_clip_loss(new_lp, old_lp, adv, eps):
    # Mean PPO clipped loss over tokens. new_lp, old_lp: log-probs of the sampled tokens; adv: advantage per token.
    ratio = np.exp(np.asarray(new_lp, dtype=float) - np.asarray(old_lp, dtype=float))
    adv = np.asarray(adv, dtype=float)
    return float(np.mean(np.maximum(-adv * ratio, -adv * np.clip(ratio, 1 - eps, 1 + eps))))

def ppo_clip_grad(new_lp, old_lp, adv, eps):
    # Gradient of the mean loss with respect to each new_lp. Zero where the clip is active.
    ratio = np.exp(np.asarray(new_lp, dtype=float) - np.asarray(old_lp, dtype=float))
    adv = np.asarray(adv, dtype=float)
    active = ((ratio > 1 + eps) & (adv > 0)) | ((ratio < 1 - eps) & (adv < 0))
    return np.where(active, 0.0, -adv * ratio) / len(ratio)

def grpo_advantages(rewards, G, scale=True):
    # Group-relative advantages: rewards holds G replies per prompt, prompt after prompt. Bessel-corrected std, plus 1e-4.
    r = np.asarray(rewards, dtype=float).reshape(-1, G)
    a = r - r.mean(1, keepdims=True)
    if scale:
        a = a / (r.std(1, ddof=1, keepdims=True) + 1e-4)
    return a.reshape(-1)

# Checks. Run the cell; it prints All checks passed when all three functions are right.
assert abs(ppo_clip_loss([0.0, 0.0], [0.0, 0.0], [1.0, -3.0], 0.2) - 1.0) < 1e-12, "ratio 1: the loss is -mean(adv)"
assert abs(ppo_clip_loss([np.log(1.5)], [0.0], [1.0], 0.2) + 1.2) < 1e-12, "a good token cannot earn more than (1 + eps) x adv"
assert abs(ppo_clip_loss([np.log(1.5)], [0.0], [-1.0], 0.2) - 1.5) < 1e-12, "a bad token that became more likely is not clipped"
assert abs(ppo_clip_loss([np.log(0.5)], [0.0], [-1.0], 0.2) - 0.8) < 1e-12, "a bad token cannot earn more than (1 - eps) x |adv|"
x, o, a = np.array([0.3, -0.4, 0.05, 0.25]), np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, -2.0, 0.5, -1.0])
g = ppo_clip_grad(x, o, a, 0.2)
num = [(ppo_clip_loss(x + 1e-6 * np.eye(4)[i], o, a, 0.2) - ppo_clip_loss(x - 1e-6 * np.eye(4)[i], o, a, 0.2)) / 2e-6 for i in range(4)]
assert np.allclose(g, num, atol=1e-6), (g, num)
assert g[0] == 0 and g[1] == 0, "both clipped tokens get no gradient"
adv = grpo_advantages([1, 0, 0, 1, 1, 1, 1, 1], 4)
assert np.allclose(adv[:4], [0.8660, -0.8660, -0.8660, 0.8660], atol=1e-3), adv[:4]
assert np.all(adv[4:] == 0), "a group where every reply scored the same gives zero advantage"
assert np.allclose(grpo_advantages([1, 0, 0, 0], 4, scale=False), [0.75, -0.25, -0.25, -0.25])
print("loss", round(ppo_clip_loss(x, o, a, 0.2), 4), "| grad", np.round(g, 4), "| advantages", np.round(adv, 3))
print("All checks passed")
