dc, dr = final["dc"], final["dr"]
nc = np.array([len(h["c"]) - len(h["p"]) for h in held])
nr = np.array([len(h["r"]) - len(h["p"]) for h in held])
print("held-out pairs", len(held), "| chosen log-prob fell in", round(float((dc < 0).mean()), 3), "| rejected fell in", round(float((dr < 0).mean()), 3),
      "| both fell in", round(float(((dc < 0) & (dr < 0)).mean()), 3), "| margin grew in", round(float((dc > dr).mean()), 3))
print("mean change per reply token, chosen", round(float((dc / nc).mean()), 4), "| rejected", round(float((dr / nr).mean()), 4))
print("correlation between the chosen and rejected change of a pair", round(float(np.corrcoef(dc, dr)[0, 1]), 3))
for h in runs.get("dpo-b0.1-s0", {}).get("held", []):
    print("step", h["step"], "| chosen", h["chosen"], "| rejected", h["rejected"], "| margin", h["margin"], "| accuracy", h["accuracy"],
          "| chosen fell", h["chosenDown"])
