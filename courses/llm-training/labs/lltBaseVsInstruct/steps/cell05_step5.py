from collections import Counter
from prompts import PROMPTS          # 40 prompts in five groups of 8, written by the notebook's first cell

# The long cell of the lab: 40 greedy answers, each scored by both models.
USE_OUR_ANSWERS = False   # True: load our 40 answers and their per-token scores from the Run Library instead
OUR_RUN = "https://raw.githubusercontent.com/genieincodebottle/aiml-companion/main/courses/llm-training/runs/lltBaseVsInstruct/lltBaseVsInstruct-t1-smollm2-kl-v2026-11.json"
ours = {}
if USE_OUR_ANSWERS:
    import json, urllib.request
    try:
        ours = {s["prompt"]: s for s in json.load(urllib.request.urlopen(OUR_RUN, timeout=60))["samples"]}
    except OSError as err:
        raise RuntimeError("Our run file is not reachable (" + str(err) + "). Set USE_OUR_ANSWERS = False to compute the answers.") from err

rows, responses = [], {}
for p in PROMPTS:
    if p in ours:                                    # our answer, with our KL and base rank for each token
        resp = ours[p]["responseIds"]
        kl, rank = [x[1] for x in ours[p]["perToken"]], [x[2] for x in ours[p]["perToken"]]
    else:
        ctx = chat(p)
        resp = greedy(inst, tok_i, ctx, 128)
        kl, rank = kl_and_rank(logprobs(inst, ctx, resp), logprobs(base, ctx, resp), resp)
    responses[p] = resp
    for t, r in enumerate(resp):
        if r != tok_i.eos_token_id:                  # the end token is scored on its own below
            rows.append((t + 1, float(kl[t]), int(rank[t]), tok_i.decode([r]).strip() or repr(tok_i.decode([r]))))

pos = np.array([r[0] for r in rows]); kl_all = np.array([r[1] for r in rows]); rk = np.array([r[2] for r in rows])
print(len(rows), "response tokens from", len(PROMPTS), "prompts")
print("unshifted, base rank 1  ", round(float((rk == 1).mean()), 3))
print("marginal, base rank 2-3 ", round(float(((rk > 1) & (rk <= 3)).mean()), 3))
print("shifted, base rank > 3  ", round(float((rk > 3).mean()), 3))
for lo, hi in [(1, 1), (2, 5), (6, 20), (21, 60), (61, 128)]:
    m = (pos >= lo) & (pos <= hi)
    print("positions", lo, "to", hi, "|", round(float(m.mean()), 3), "of tokens |",
          round(float(kl_all[m].sum() / kl_all.sum()), 3), "of the KL")
print(Counter(r[3] for r in rows if r[2] > 3).most_common(12))
