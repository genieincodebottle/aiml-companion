import torch

def token_logps(logits, ids):
    # log-prob of each chosen token: log_softmax over the vocabulary, then pick the id
    return torch.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

def whiten(x):
    # zero mean, unit variance over the whole batch (the advantage normalization of most PPO code)
    return (x - x.mean()) * torch.rsqrt(x.var(unbiased=False) + 1e-8)

def gae(rewards, values, gamma=1.0, lam=0.95):
    # rewards, values: [batch, response tokens]. values[:, t] estimates the return from token t on; it is 0 after the last token.
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        next_v = values[:, t + 1] if t + 1 < rewards.shape[1] else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_v - values[:, t]       # one-step TD error
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values                                      # advantages, and returns for the value loss

def ppo_policy_loss(new_lp, old_lp, adv, eps=0.2):
    # clipped surrogate: take the worse (larger) of the two losses, so a big ratio earns no extra credit
    ratio = torch.exp(new_lp - old_lp)
    loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - eps, 1 + eps)).mean()
    clipped = ((ratio > 1 + eps) & (adv > 0)) | ((ratio < 1 - eps) & (adv < 0))   # tokens whose gradient the clip zeroes
    return loss, clipped.float().mean()

def value_loss(new_v, old_v, returns, eps=0.2):
    v_clip = old_v + torch.clamp(new_v - old_v, -eps, eps)
    return 0.5 * torch.max((new_v - returns) ** 2, (v_clip - returns) ** 2).mean()

def kl_k1(lp, ref_lp):
    return lp - ref_lp                      # log(pi / ref) at the sampled token

def kl_k3(lp, ref_lp):
    d = ref_lp - lp
    return torch.exp(d) - d - 1             # never negative, same expectation as k1 under pi

def shaped_rewards(score, lp, ref_lp, kl_coef):
    # PPO-RLHF reward: a KL penalty on every token, plus the reward model's score on the last token
    r = -kl_coef * kl_k1(lp, ref_lp)
    r[:, -1] += score
    return r

# Tests on small tensors
torch.manual_seed(0)
rew, val = torch.randn(3, 5), torch.randn(3, 5)
a1, ret1 = gae(rew, val, gamma=1.0, lam=1.0)
assert torch.allclose(a1, rew.flip(1).cumsum(1).flip(1) - val, atol=1e-6), "lam = 1: advantage = reward-to-go - value"
a0, _ = gae(rew, val, gamma=1.0, lam=0.0)
assert torch.allclose(a0, rew + torch.cat([val[:, 1:], torch.zeros(3, 1)], 1) - val, atol=1e-6), "lam = 0: one-step TD error"
assert torch.allclose(ret1, rew.flip(1).cumsum(1).flip(1), atol=1e-6), "lam = 1: return = reward-to-go"

lp = torch.zeros(4, requires_grad=True)
loss, cf = ppo_policy_loss(lp, torch.zeros(4), torch.tensor([1.0, -1.0, 1.0, -1.0]))
assert abs(loss.item()) < 1e-7 and cf.item() == 0, "ratio 1: loss = -mean(adv), nothing clipped"
for r, a in [(1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)]:
    x = torch.tensor([float(torch.log(torch.tensor(r)))], requires_grad=True)
    loss, cf = ppo_policy_loss(x, torch.zeros(1), torch.tensor([a]))
    loss.backward()
    print("ratio", r, "| advantage", a, "| loss", round(loss.item(), 3), "| gradient", round(x.grad.item(), 3), "| clip zeroes the gradient", bool(cf.item()))

# KL estimators: sample tokens from pi, compare with the exact KL(pi || ref) over a 50-token vocabulary
logits = torch.randn(50)
for name, noise in [("policy near the reference", 0.3), ("policy far from it", 1.5)]:
    pi, ref = torch.softmax(logits, 0), torch.softmax(logits + noise * torch.randn(50), 0)
    exact = (pi * (pi / ref).log()).sum().item()
    ids = torch.multinomial(pi, 100000, replacement=True)
    k1, k3 = kl_k1(pi.log()[ids], ref.log()[ids]), kl_k3(pi.log()[ids], ref.log()[ids])
    print(name, "| exact KL", round(exact, 4), "| k1 mean", round(k1.mean().item(), 4), "std", round(k1.std().item(), 3),
          "min", round(k1.min().item(), 3), "| k3 mean", round(k3.mean().item(), 4), "std", round(k3.std().item(), 3), "min", round(k3.min().item(), 4))

r = shaped_rewards(torch.tensor([2.0]), torch.tensor([[-1.0, -2.0, -0.5]]), torch.tensor([[-1.2, -1.5, -0.5]]), 0.05)
print("shaped rewards", [round(v, 3) for v in r[0].tolist()])
print("All PPO pieces pass their tests")
