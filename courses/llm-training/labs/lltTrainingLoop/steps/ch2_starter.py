import numpy as np


def mean_se(xs):
    # TODO: return the mean and its standard error, the sample std (ddof=1) over the square root of n
    return None, None


def is_real(a, b):
    # TODO: return (difference, SE of the difference, verdict).
    # difference = mean(b) - mean(a). The SE of a difference of two independent means is
    # the square root of the sum of their squared SEs. verdict is True when |difference| > 2 SE.
    return None, None, None


# Final val loss of the 10.77M model, 3 seeds per learning rate, from our T4 runs
lr_3e4 = [1.5405567, 1.5464902, 1.5386005]
lr_1e3 = [1.4505350, 1.4559186, 1.4601615]
lr_3e3 = [1.4569817, 1.4595672, 1.4576760]
# 7 more seeds at 1e-3, so 10 in all
lr_1e3_10 = lr_1e3 + [1.4509572, 1.4445310, 1.4488465, 1.4616865, 1.4480715, 1.4485415, 1.4480330]

for name, a, b in [("3e-4 vs 1e-3", lr_3e4, lr_1e3), ("1e-3 vs 3e-3", lr_1e3, lr_3e3), ("1e-3 x10 vs 3e-3", lr_1e3_10, lr_3e3)]:
    d, se, real = is_real(a, b)
    print(name, "| difference", d if d is None else round(d, 4), "| SE", se if se is None else round(se, 4), "| real", real)

# Checks
m, e = mean_se(lr_1e3)
assert m is not None and abs(m - 1.45554) < 1e-4, "mean_se: the mean of lr_1e3 should be about 1.4555"
assert abs(e - 0.00279) < 1e-4, "mean_se: the SE should use ddof=1, about 0.0028 for lr_1e3"
d, se, real = is_real(lr_3e4, lr_1e3)
assert abs(d + 0.0863) < 1e-3 and real is True, "3e-4 vs 1e-3 is a difference of -0.0863, about 24 SE, so it is real"
d, se, real = is_real(lr_1e3, lr_3e3)
assert abs(d - 0.0025) < 1e-3 and abs(se - 0.0029) < 2e-4 and real is False, "1e-3 vs 3e-3 is 0.0025 with an SE of 0.0029, under 1 SE"
d, se, real = is_real(lr_1e3_10, lr_3e3)
assert abs(d - 0.0063) < 2e-4 and abs(se - 0.0019) < 2e-4 and real is True, "with 10 seeds at 1e-3 the gap is 0.0063, about 3.3 SE, so it is real"
print("All checks passed")
