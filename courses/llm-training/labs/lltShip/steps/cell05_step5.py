RUN_GRID = False      # True scores 9 more merges (each panel took about 1 min on our A100 and 2 min on our T4). With FROM_OUR_CHECKPOINT on, only our best merge is scored

def kept(m):
    # Share of each fine-tune's gain the merge keeps: 1.0 = as good as that fine-tune, 0 = no better than the base
    hi = (runs["base"]["hiBpb"] - m["hiBpb"]) / (runs["base"]["hiBpb"] - runs["hindi"]["hiBpb"])
    rep = (runs["base"]["replyNll"] - m["replyNll"]) / (runs["base"]["replyNll"] - runs["sft"]["replyNll"])
    return round(float(hi), 3), round(float(rep), 3)

GRID = [("average, lambda 0.5", "ta", dict(lams=(0.5, 0.5))),
        ("task arithmetic 1.0", "ta", dict(lams=(1.0, 1.0))),
        ("TIES density 0.2", "ties", dict(density=0.2))]
if RUN_GRID:
    GRID += [("task arithmetic 0.75", "ta", dict(lams=(0.75, 0.75))),
             ("task arithmetic 1.25", "ta", dict(lams=(1.25, 1.25))),
             ("task arithmetic 1.0 / 0.5", "ta", dict(lams=(1.0, 0.5))),       # (lambda for SFT, lambda for Hindi)
             ("task arithmetic 1.0 / 0.25", "ta", dict(lams=(1.0, 0.25))),
             ("TIES density 0.5", "ties", dict(density=0.5)),
             ("TIES density 1.0", "ties", dict(density=1.0)),
             ("DARE p 0.5", "dare", dict(p=0.5, seed=0)),
             ("DARE p 0.9", "dare", dict(p=0.9, seed=0)),
             ("DARE p 0.9 seed 1", "dare", dict(p=0.9, seed=1))]
elif FROM_OUR_CHECKPOINT:
    # Score only the merge we shipped, and read the other eight from our grid, so the rule and every later step match the page
    GRID.append(("task arithmetic 1.0 / 0.5", "ta", dict(lams=(1.0, 0.5))))
    for n, m in REC["grid"].items():
        if n not in [g[0] for g in GRID]:
            print((n.ljust(26) + " | Hindi bpb " + format(m["hiBpb"], ".4f") + " | reply NLL " + format(m["replyNll"], ".4f")
                   + " | kept " + str(m["kept"])), "| recorded on our A100")
for name, method, kw in GRID:
    t0 = time.time()
    sd = merge_model(method, **kw)
    t1 = time.time()
    load_into(sd)
    runs[name] = panel(model)
    runs[name]["kept"] = kept(runs[name])
    runs[name]["mergeSec"], runs[name]["panelSec"] = round(t1 - t0, 1), round(time.time() - t1, 1)
    show(name, runs[name])
    print("   kept (Hindi, replies)", runs[name]["kept"], "| merge", runs[name]["mergeSec"], "s on the CPU, panel",
          runs[name]["panelSec"], "s")

# Pick the merge whose weaker skill keeps the most of its gain, then save it for parts 2 and 3
best = max((n for n, _, _ in GRID), key=lambda n: min(runs[n]["kept"]))
print("ship:", best, "| Hindi bpb vs hindi model", ci(runs["hindi"], runs[best], "hi"),
      "| reply NLL vs sft model", ci(runs["sft"], runs[best], "reply"), "| English bpb vs base", ci(runs["base"], runs[best], "en"))
method, kw = {n: (m, k) for n, m, k in GRID}[best]
load_into(merge_model(method, **kw))
model.save_pretrained("merged")
tok.chat_template = ("{% for m in messages %}{% if m['role'] == 'user' %}### Instruction:" + NL + "{{ m['content'] }}" + NL + NL
                     + "### Response:" + NL + "{% elif m['role'] == 'assistant' %}{{ m['content'] }}<|endoftext|>{% endif %}{% endfor %}")
tok.save_pretrained("merged")
print(sorted(os.listdir("merged")))
