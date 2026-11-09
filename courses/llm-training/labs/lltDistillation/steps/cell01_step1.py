import numpy as np

x = np.linspace(-6, 6, 1201)
dx = x[1] - x[0]

def normal(mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def kl(a, b):
    # KL(a || b) on the grid, in nats. The tiny floor stops log(0) where a density underflows.
    return float(np.sum(a * (np.log(a + 1e-300) - np.log(b + 1e-300))) * dx)

teacher = 0.6 * normal(-2.0, 0.6) + 0.4 * normal(2.0, 0.6)       # two modes, 60% and 40% of the mass
grid = [(mu, sd) for mu in np.arange(-4, 4.001, 0.05) for sd in np.arange(0.2, 4.001, 0.05)]
fits = {"forward KL(teacher || student)": min(grid, key=lambda g: kl(teacher, normal(*g))),
        "reverse KL(student || teacher)": min(grid, key=lambda g: kl(normal(*g), teacher))}

valley = np.abs(x) < 0.5                                          # the gap between the modes, where the teacher has little mass
print("teacher mass in the valley |x| < 0.5:", round(float(teacher[valley].sum() * dx), 4))
for name, (mu, sd) in fits.items():
    s = normal(mu, sd)
    print(name, "| best Gaussian: mean", round(mu, 2), "sd", round(sd, 2),
          "| student mass in the valley", round(float(s[valley].sum() * dx), 4),
          "| forward KL", round(kl(teacher, s), 3), "| reverse KL", round(kl(s, teacher), 3))
print("reverse KL with the student on the other mode, mean 2.0 sd 0.6:", round(kl(normal(2.0, 0.6), teacher), 3))