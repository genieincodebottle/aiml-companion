final = {"cosine": [], "wsd": []}
for schedule in final:
    for seed in range(3):
        _, hist = train(dict(CPU_CFG, schedule=schedule, seed=seed))
        final[schedule].append(hist[-1]["valLoss"])
        print(schedule, "seed", seed, "| final val", round(hist[-1]["valLoss"], 4))
