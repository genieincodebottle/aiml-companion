print("lr       runs   mean     std")
for lr in LRS:
    runs = [r["final"] for r in results if r["lr"] == lr]
    xs = [x for x in runs if x == x]          # drop NaN, a run that diverged
    if not runs:
        print(f"{lr:.0e}   0")
    elif not xs:
        print(f"{lr:.0e}   {len(runs)}      every run diverged")
    else:
        std = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
        print(f"{lr:.0e}   {len(xs)}      {np.mean(xs):.4f}   {std:.4f}")
