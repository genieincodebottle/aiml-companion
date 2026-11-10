after = evaluate_gsm(grpo_model, 8 if PASS_AT_K else 0)
print("after GRPO", {k: v for k, v in after.items() if k != "replies"})
print("before    ", {k: v for k, v in base_eval.items() if k != "replies"})
right = [i for i, (a, b) in enumerate(zip(base_eval["replies"], after["replies"])) if correct(b, test[i]["gold"]) and not correct(a, test[i]["gold"])]
if right:
    i = right[0]
    print("question", i, "gold", test[i]["gold"])
    print("before:", repr(cut(base_eval["replies"][i])[:300]))
    print("after: ", repr(cut(after["replies"][i])[:300]))
