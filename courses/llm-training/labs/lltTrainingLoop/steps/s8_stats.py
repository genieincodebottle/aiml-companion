def mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    return xs.mean(), xs.std(ddof=1), xs.std(ddof=1) / np.sqrt(len(xs))


def compare(a, b, name_a, name_b):
    ma, sa, ea = mean_se(a)
    mb, sb, eb = mean_se(b)
    diff, se_diff = mb - ma, np.sqrt(ea ** 2 + eb ** 2)
    print(f"{name_a}: mean {ma:.4f}, std {sa:.4f}, SE {ea:.4f}")
    print(f"{name_b}: mean {mb:.4f}, std {sb:.4f}, SE {eb:.4f}")
    verdict = "a real difference" if abs(diff) > 2 * se_diff else "not a claim, within 2 SE"
    print(f"difference {diff:+.4f} | SE of difference {se_diff:.4f} | {abs(diff) / se_diff:.1f} SE | {verdict}")


compare(final["cosine"], final["wsd"], "cosine", "wsd")
