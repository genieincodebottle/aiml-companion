msgs = [{"role": "user", "content": "What is the capital of France?"}]
print(tok_i.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

enc = tok_i.apply_chat_template(msgs, add_generation_prompt=True)
ids = list(enc["input_ids"])
print(type(enc).__name__, list(enc.keys()), len(ids), "tokens")
print(tok_i.convert_ids_to_tokens(ids)[-8:])
no_gen = tok_i.apply_chat_template(msgs, add_generation_prompt=False)
print(len(no_gen["input_ids"]), "tokens without the generation prompt")

print("instruct EOS", repr(tok_i.eos_token), tok_i.eos_token_id, "| base EOS", repr(tok_b.eos_token), tok_b.eos_token_id)
print("base ships a chat template:", tok_b.chat_template is not None)
print("same vocabulary:", tok_b.get_vocab() == tok_i.get_vocab(), len(tok_i), "tokens")
