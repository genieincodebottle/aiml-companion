def chat(prompt):
    msgs = [{"role": "user", "content": prompt}]
    return list(tok_i.apply_chat_template(msgs, add_generation_prompt=True)["input_ids"])

for prompt in THREE:
    ids = chat(prompt)
    print(prompt)
    for name, tok, model in [("base", tok_b, base), ("instruct", tok_i, inst)]:
        print("  ", name.ljust(9), repr(tok.decode(greedy(model, tok, ids, 40))[:90]))
