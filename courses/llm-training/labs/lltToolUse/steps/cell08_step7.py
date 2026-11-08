from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MAIN, dtype=torch.float16).to("cuda").eval()
t0 = time.time()
base_out = C.generate(model, tok, items, batch_size=32, max_new_tokens=384)
base_sec = round(time.time() - t0)
base = C.score(items, base_out, parse_xml_calls)
print("base model |", base_sec, "s |", json.dumps(C.summary(base), indent=1))
print("agreement with BFCL's checker:", sum(official(it, r["calls"]) == r["correct"] for it, r in zip(items, base)), "of", len(items))
print(base_out[0][:300])
