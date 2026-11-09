import numpy as np

def dpo_loss(pc, pr, rc, rr, beta):
    # Mean DPO loss over pairs. pc, pr: policy log-probs of chosen and rejected replies; rc, rr: reference log-probs.
    raise NotImplementedError

def dpo_grads(pc, pr, rc, rr, beta):
    # Gradient of the mean DPO loss with respect to each pc and each pr. Return (d_pc, d_pr) as arrays.
    raise NotImplementedError

def ipo_loss(pc, pr, rc, rr, nc, nr, beta):
    # Mean IPO loss in TRL's form: per-token log-ratio gap, squared distance to 1 / (2 beta).
    raise NotImplementedError

def simpo_loss(pc, pr, nc, nr, beta, gamma):
    # Mean SimPO loss: no reference, per-token log-probs, target margin gamma.
    raise NotImplementedError

# Checks. Run the cell; it prints All checks passed when all four functions are right.
rc, rr = np.array([-120.0, -80.0]), np.array([-135.0, -95.0])
assert abs(dpo_loss(rc, rr, rc, rr, 0.1) - np.log(2)) < 1e-12, "policy = reference gives ln 2"
assert abs(dpo_loss(rc - 10, rr - 20, rc, rr, 0.1) - dpo_loss(rc + 5, rr - 5, rc, rr, 0.1)) < 1e-12, "only the log-ratio margin matters"
assert abs(dpo_loss([1000.0], [0.0], [0.0], [0.0], 1.0)) < 1e-9 and abs(dpo_loss([0.0], [1000.0], [0.0], [0.0], 1.0) - 1000.0) < 1e-9, "use a stable form"
pc, pr = np.array([-118.0, -85.0]), np.array([-140.0, -90.0])
gc_, gr_ = dpo_grads(pc, pr, rc, rr, 0.1)
eps = 1e-5
num = [(dpo_loss(pc + eps * np.eye(2)[i], pr, rc, rr, 0.1) - dpo_loss(pc - eps * np.eye(2)[i], pr, rc, rr, 0.1)) / (2 * eps) for i in range(2)]
assert np.allclose(gc_, num, atol=1e-8), (gc_, num)
assert np.allclose(gc_ + gr_, 0), "the chosen and rejected gradients are equal and opposite"
assert abs(ipo_loss(rc, rr, rc, rr, [100, 50], [80, 60], 0.1) - 25.0) < 1e-12, "IPO starts at (1 / (2 beta)) squared"
assert abs(ipo_loss(rc - 10, rr - 40, rc, rr, [100, 50], [80, 60], 5.0) - 0.112222222222) < 1e-9, "divide each log-ratio by its reply length"
assert abs(simpo_loss([-100.0], [-100.0], [100], [50], 2.0, 1.0) - np.logaddexp(0, -(2.0 * (-1.0 + 2.0) - 1.0))) < 1e-12
print("DPO loss", round(dpo_loss(pc, pr, rc, rr, 0.1), 4), "| d_pc", np.round(gc_, 5), "| IPO", round(ipo_loss(pc, pr, rc, rr, [100, 50], [80, 60], 0.1), 4),
      "| SimPO", round(simpo_loss(pc, pr, [100, 50], [80, 60], 2.0, 1.0), 4))
print("All checks passed")
