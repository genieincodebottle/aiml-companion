import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bt_loss(r_chosen, r_rejected):
    # -log sigmoid(r_chosen - r_rejected), averaged over pairs. logaddexp(0, -d) never overflows.
    d = np.asarray(r_chosen, dtype=float) - np.asarray(r_rejected, dtype=float)
    return float(np.mean(np.logaddexp(0, -d)))

def bt_grad(r_chosen, r_rejected):
    # gradient of the mean loss with respect to each chosen and each rejected reward
    d = np.asarray(r_chosen, dtype=float) - np.asarray(r_rejected, dtype=float)
    g = -(1 - sigmoid(d)) / len(d)
    return g, -g

print("P(chosen preferred) at margin 0.5:", round(float(sigmoid(0.5)), 4), "| loss", round(bt_loss([0.5], [0.0]), 4))
print("loss at margin 0:", round(bt_loss([0, 0], [0, 0]), 4), "| ln 2 =", round(float(np.log(2)), 4))
print("add 7 to both rewards:", round(bt_loss([7.5], [7.0]), 4), "| scale the margin by 4:", round(bt_loss([2.0], [0.0]), 4))

rc, rr = np.array([1.2, -0.3, 0.4]), np.array([0.2, 0.5, 0.4])
gc, gr = bt_grad(rc, rr)
eps = 1e-6
num = [(bt_loss(rc + eps * np.eye(3)[i], rr) - bt_loss(rc - eps * np.eye(3)[i], rr)) / (2 * eps) for i in range(3)]
print("analytic dL/dr_chosen", np.round(gc, 5), "| finite difference", np.round(num, 5))
print("gradient sums to zero per pair:", bool(np.allclose(gc + gr, 0)))
