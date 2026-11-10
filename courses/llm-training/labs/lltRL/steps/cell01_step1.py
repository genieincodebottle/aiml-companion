import numpy as np

# One prompt, four possible replies. The policy is a softmax over four logits; each reply has a fixed reward.
reward = np.array([0.0, 1.0, 2.0, 5.0])
theta = np.zeros(4)

def probs(theta):
    e = np.exp(theta - theta.max())
    return e / e.sum()

def exact_grad(theta, reward):
    # d/dtheta of E[reward] = sum over replies of p(y) * reward(y) * grad log p(y); grad log p(y) = onehot(y) - p
    p = probs(theta)
    return p * (reward - p @ reward)

def reinforce(theta, reward, n, baseline, rng):
    # REINFORCE: sample n replies, weight grad log p(y) by (reward - baseline), average
    p = probs(theta)
    ys = rng.choice(len(p), size=n, p=p)
    return ((reward[ys] - baseline)[:, None] * (np.eye(len(p))[ys] - p)).mean(0)

rng = np.random.default_rng(0)
print("exact gradient", np.round(exact_grad(theta, reward), 4))
for shift in [0.0, 100.0]:
    r = reward + shift
    for name, b in [("no baseline", 0.0), ("baseline = mean reward", probs(theta) @ r)]:
        est = np.array([reinforce(theta, r, 8, b, rng) for _ in range(2000)])
        print("rewards +" + str(int(shift)), "|", name, "| mean of 2,000 estimates", np.round(est.mean(0), 3),
              "| spread (std, summed over the 4 logits)", round(float(est.std(0).sum()), 3))

# Train with 8 samples per step and the batch mean as the baseline, as GRPO does with a group
theta = np.zeros(4)
for step in range(201):
    p = probs(theta)
    if step % 50 == 0:
        print("step", step, "| p(best reply)", round(float(p[3]), 3), "| expected reward", round(float(p @ reward), 3))
    ys = rng.choice(4, size=8, p=p)
    adv = reward[ys] - reward[ys].mean()
    theta += 0.5 * (adv[:, None] * (np.eye(4)[ys] - p)).mean(0)
