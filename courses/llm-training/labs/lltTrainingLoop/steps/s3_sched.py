from torch.optim.lr_scheduler import LambdaLR


def lr_cosine(step, peak, warmup, total, min_ratio=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def lr_wsd(step, peak, warmup, total, min_ratio=0.1, decay_frac=0.2):
    # Warmup, then a flat stable phase, then a linear decay over the last decay_frac of steps.
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    progress = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (1 - (1 - min_ratio) * progress)


def make_scheduler(opt, cfg):
    fn = lr_wsd if cfg["schedule"] == "wsd" else lr_cosine
    # LambdaLR multiplies the optimizer's base lr (cfg["lr"]) by this factor at every step.
    return LambdaLR(opt, lambda s: fn(s, cfg["lr"], cfg["warmup"], cfg["steps"]) / cfg["lr"])
