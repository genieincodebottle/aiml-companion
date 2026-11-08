t0 = time.time()
sft_forced_out = C.generate(model, tok, items, constrained=True)
with model.disable_adapter():                      # the same weights without the adapters: the base model
    base_forced_out = C.generate(model, tok, items, constrained=True)
forced_sec = round(time.time() - t0)
sft_forced = C.score(items, sft_forced_out, parse_xml_calls)
base_forced = C.score(items, base_forced_out, parse_xml_calls)
print("constrained decoding, both models |", forced_sec, "s")
for name, r in [("base", base), ("base, constrained", base_forced), ("LoRA", sft), ("LoRA, constrained", sft_forced)]:
    s = C.summary(r)
    print(name.ljust(18), "| parsed", s["parseRate"], "| correct", s["accuracy"],
          "| irrelevance", s["byCategory"]["irrelevance"]["accuracy"], "| simple", s["byCategory"]["simple_python"]["accuracy"])
