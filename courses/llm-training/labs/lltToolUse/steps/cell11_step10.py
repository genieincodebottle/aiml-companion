from transformers import AutoModelForCausalLM

SPECIAL = ["<tool_call>", "</tool_call>", "<|im_end|>", "<think>", "</think>"]
ref_check = AutoModelForCausalLM.from_pretrained(REF, dtype=torch.float16)      # on the CPU, to read its embedding rows
E = ref_check.get_input_embeddings().weight.float()
norms = E.norm(dim=1)
print("Qwen3-1.7B-Base | median row norm", round(norms.median().item(), 4),
      "|", {s: round(norms[ref_tok.convert_tokens_to_ids(s)].item(), 4) for s in SPECIAL})
del ref_check, E

def report(name, outs, rows_):
    relevant = [r for r in rows_ if r["category"] != "irrelevance"]
    print(name, "| relevant items right", sum(r["correct"] for r in relevant), "of", len(relevant),
          "| replies with a <tool_call> tag", sum("<tool_call>" in o for o in outs), "of", len(outs),
          "| accuracy", C.summary(rows_)["accuracy"])

RUN_EXPERIMENT = False      # True trains both Qwen3 arms on your T4; our numbers are in the table below
ref_results = {}
if RUN_EXPERIMENT:
    model = trainer = base_model = None            # drop every reference to the 2B model and its optimizer state
    gc.collect()
    torch.cuda.empty_cache()
    # TRL has no training template for Qwen3-Base, so render with its own template; the label rule is the same
    REF_HEADER = ref_tok.encode("<|im_start|>assistant" + C.NL, add_special_tokens=False)
    REF_END, REF_NL = ref_tok.convert_tokens_to_ids("<|im_end|>"), ref_tok.encode(C.NL, add_special_tokens=False)[0]
    ref_rows = []
    for c in train:
        ids = list(ref_tok.apply_chat_template(c["messages"], tools=c["tools"])["input_ids"])
        ref_rows.append({"input_ids": ids, "labels": build_labels(ids, REF_HEADER, REF_END, REF_NL)})
    ref_model = AutoModelForCausalLM.from_pretrained(REF, dtype=torch.float16).to("cuda").eval()
    outs = C.generate(ref_model, ref_tok, items)
    ref_results["base"] = (outs, C.score(items, outs, C.parse_json_calls))
    report("Qwen3-1.7B-Base", *ref_results["base"])
    del ref_model
    for name, special in [("LoRA", None), ("LoRA, special rows set first", SPECIAL)]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ref_trainer = train_lora(REF, ref_tok, ref_rows, seed=0, special=special)
        outs = C.generate(ref_trainer.model.eval(), ref_tok, items)
        ref_results[name] = (outs, C.score(items, outs, C.parse_json_calls))
        report(name, *ref_results[name])
        print("   peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
        ref_trainer = None
