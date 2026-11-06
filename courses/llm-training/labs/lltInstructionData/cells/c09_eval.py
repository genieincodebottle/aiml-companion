from lm_eval.tasks.ifeval.utils import process_results

ifeval_path = hf_hub_download("google/IFEval", "ifeval_input_data.jsonl", repo_type="dataset",
                              revision="966cd89545d6b6acfd7638bc708b98261ca58e84")
ifeval = [json.loads(line) for line in open(ifeval_path, encoding="utf-8")]
N_IFEVAL = int(os.environ.get("LLT_N_IFEVAL", str(len(ifeval))))

@torch.no_grad()
def answer_all(model, tok, prompts, max_new=512, batch_size=32):
    # Greedy answers in the chat format, left-padded, stopped at the end-of-turn token.
    tok.padding_side = "left"
    model.to(device)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    outs = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id,
                           pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        outs += [(tok.decode(row, skip_special_tokens=True), bool((row == tok.eos_token_id).any())) for row in new]
    return outs

def ifeval_scores(model, tok):
    docs = ifeval[:N_IFEVAL]
    outs = answer_all(model, tok, [d["prompt"] for d in docs])
    res = [process_results(d, [text]) for d, (text, _) in zip(docs, outs)]
    loose = np.array([r["prompt_level_loose_acc"] for r in res], dtype=float)
    strict = np.array([r["prompt_level_strict_acc"] for r in res], dtype=float)
    stopped = np.mean([s for _, s in outs])
    return loose, strict, stopped, [t for t, _ in outs]
