import json
from huggingface_hub import hf_hub_download

# our pool: 64 held-out prompts x 64 samples from SmolLM2-360M-Instruct, with the gold reward of each sample
pool = json.load(open(hf_hub_download("aimlcompanion/llm-training-lab", "lltRewardModels/bon_pool.json")))
P, N = len(pool["prompts"]), pool["n"]
proxy = np.array(score(rm, [chat_ids(pool["prompts"][s["p"]]["prompt"], s["text"]) for s in pool["samples"]])).reshape(P, N)
gold = np.array(pool["gold"]).reshape(P, N)
tokens = np.array([s["newTokens"] for s in pool["samples"]]).reshape(P, N)

def best_of_n(select, value, n):
    # expected value of the sample that select ranks first among n drawn without replacement from the N recorded ones
    total = 0.0
    for p in range(len(select)):
        order = np.argsort(select[p])                       # ascending: rank i has i samples below it
        weight = np.array([math.comb(i, n - 1) for i in range(N)], dtype=float) / math.comb(N, n)
        total += float((weight * value[p][order]).sum())
    return total / len(select)

print("n    KL nats   proxy    gold     new tokens")
for n in [1, 2, 4, 8, 16, 32, 64]:
    kl = math.log(n) - (n - 1) / n
    print(n, round(kl, 3), round(best_of_n(proxy, proxy, n), 3), round(best_of_n(proxy, gold, n), 3),
          round(best_of_n(proxy, tokens, n), 1), sep="    ")
print("gold picking by gold, n = 64:", round(best_of_n(gold, gold, 64), 3))
