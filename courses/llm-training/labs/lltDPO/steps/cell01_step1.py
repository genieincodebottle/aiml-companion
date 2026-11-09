import numpy as np

def dpo_loss(pw, pl, rw, rl, beta):
    # pw, pl: policy log-probs of the chosen and rejected replies; rw, rl: the same under the frozen reference
    z = beta * ((np.asarray(pw, dtype=float) - rw) - (np.asarray(pl, dtype=float) - rl))
    return float(np.mean(np.logaddexp(0, -z)))

def push(pw, pl, rw, rl, beta):
    # sigmoid(-z): the share of the full gradient a pair still gets, 0.5 at the start and near 0 once it is learned
    z = beta * ((np.asarray(pw, dtype=float) - rw) - (np.asarray(pl, dtype=float) - rl))
    return float(np.mean(1 / (1 + np.exp(z))))

rw, rl = -120.0, -135.0          # reference log-probs of one chosen and one rejected reply, summed over reply tokens
print("policy = reference | loss", round(dpo_loss(rw, rl, rw, rl, 0.1), 4), "| ln 2 =", round(float(np.log(2)), 4))
for name, pw, pl in [("chosen +5, rejected -5", rw + 5, rl - 5),
                     ("chosen -10, rejected -20", rw - 10, rl - 20),
                     ("chosen -40, rejected -50", rw - 40, rl - 50)]:
    print(name, "| loss", round(dpo_loss(pw, pl, rw, rl, 0.1), 4),
          "| implicit rewards", round(0.1 * (pw - rw), 2), "and", round(0.1 * (pl - rl), 2))
for beta in [0.01, 0.1, 0.5]:
    print("beta", beta, "| margin of 10 nats: loss", round(dpo_loss(rw - 10, rl - 20, rw, rl, beta), 4),
          "| push left", round(push(rw - 10, rl - 20, rw, rl, beta), 3))
print("IPO at the start, beta 0.1: (0 - 1 / (2 * 0.1)) squared =", round((0 - 1 / (2 * 0.1)) ** 2, 4))
