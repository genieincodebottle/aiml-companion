USE_OUR_TEACHER_DATA = True    # False: the teacher writes all 2,048 training answers on your T4 instead of downloading ours

if USE_OUR_TEACHER_DATA:
    from huggingface_hub import hf_hub_download
    ours = json.load(open(hf_hub_download("aimlcompanion/llm-training-lab", "lltDistillation/teacher_train.json")))
    assert ours["questions"] == [x["q"] for x in train], "the shuffle differs from ours; keep seed=0 and N_TRAIN = 2048"
    teacher_train = ours["completionIds"]
else:
    t0 = time.time()
    teacher_train = generate(teacher, [prompt(x["q"]) for x in train], batch=64)
    sec = time.time() - t0
    new_tokens = sum(len(c) for c in teacher_train)
    print("teacher wrote", new_tokens, "tokens in", round(sec), "s |", round(new_tokens / sec), "tokens/s")
    json.dump({"questions": [x["q"] for x in train], "completionIds": teacher_train}, open("teacher_train.json", "w"))

texts = [tok.decode(c, skip_special_tokens=True) for c in teacher_train]
right = [extract(t) is not None and extract(t) == gold(x["a"]) for t, x in zip(texts, train)]
print("teacher answers", len(teacher_train), "| correct", round(float(np.mean(right)), 3), "| ended at the stop string",
      round(float(np.mean([cut(t) != t for t in texts])), 3), "| mean tokens", round(float(np.mean([len(c) for c in teacher_train])), 1))

# the human-written GSM8K solutions, given the same ending so every training target stops the same way
ref_train = [tok(" " + x["a"] + STOPS[0] + ":")["input_ids"] for x in train]
print("reference solutions, mean tokens", round(float(np.mean([len(c) for c in ref_train])), 1))
print(texts[0])