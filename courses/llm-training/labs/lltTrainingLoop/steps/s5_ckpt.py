import random


def save_checkpoint(path, model, opt, sched, scaler, sampler, step, cfg):
    state = {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(), "sampler": sampler.state_dict(), "step": step, "cfg": cfg,
        "chars": chars,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # a session that dies mid-save leaves the old file intact


def load_checkpoint(path, model, opt, sched, scaler, sampler, skip=()):
    # weights_only=False because the RNG states are Python and NumPy objects. Load only your own files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if "optimizer" not in skip:
        opt.load_state_dict(state["optimizer"])
    if "scheduler" not in skip:
        sched.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if "sampler" not in skip:
        sampler.load_state_dict(state["sampler"])
    if "rng" not in skip:
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["step"]
