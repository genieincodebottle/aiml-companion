"""Read a build result and say who noticed the break. Pure standard library,
because the notebook embeds this file as it is.

A build result is a dict with `run_ok`, `tests` (each with label, tier, status,
failures) and `score` (the scorer's revenue and shipping summaries).
"""

TIERS = ["generic", "singular", "added"]


def tiers_firing(result):
    out = {t: [] for t in TIERS}
    for t in result["tests"]:
        if t["status"] != "pass":
            out[t["tier"]].append(t["label"])
    return out


def classify(result):
    """The first line of defence that notices the break."""
    if not result["run_ok"]:
        return "build error"
    fired = tiers_firing(result)
    if fired["generic"]:
        return "generic"
    if fired["singular"]:
        return "singular only"
    if fired["added"]:
        return "added only"
    return "nothing"


def is_silent(result):
    """True when the typical suite (generic plus singular) passes and the answer is still wrong."""
    fired = tiers_firing(result)
    wrong = not (result["score"]["revenue"]["exact"] and result["score"]["shipping"]["exact"])
    return wrong and not fired["generic"] and not fired["singular"]
