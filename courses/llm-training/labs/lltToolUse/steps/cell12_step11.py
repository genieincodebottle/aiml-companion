def make_negatives(rows, start, count, shift):
    # Pair each request with the tools of another conversation that share no function name with its own tools.
    # The target reply is plain text with no call: what BFCL's irrelevance items ask for.
    out = []
    for k in range(start, start + count):
        a, b = rows[k], rows[(k - shift) % len(rows)]
        own = {t["function"]["name"] for t in a["tools"]}
        other = {t["function"]["name"] for t in b["tools"]}
        if own & other:
            continue
        out.append({"messages": [a["messages"][0], {"role": "assistant", "content": C.NO_TOOL_REPLY}], "tools": b["tools"]})
    return out

third = N_PER_SOURCE // 3
negatives = make_negatives(train, 2 * third, third, N_PER_SOURCE // 2)     # the last third of the single-turn rows
mixed = train[:2 * third] + negatives + train[N_PER_SOURCE:]
print(len(negatives), "requests with no fitting tool |", len(mixed), "conversations in the mixed set")
print(negatives[0]["messages"][0]["content"][:100], "->", [t["function"]["name"] for t in negatives[0]["tools"]])

RUN_EXPERIMENT = False      # True trains on the mixed set on your T4; our numbers are in the table below
if RUN_EXPERIMENT:
    model = trainer = ref_trainer = base_model = None
    gc.collect()
    torch.cuda.empty_cache()
    mixed_rows = []
    for c in mixed:
        ids = render(c)
        mixed_rows.append({"input_ids": ids, "labels": build_labels(ids, HEADER, IM_END, NEWLINE)})
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mix_trainer = train_lora(MAIN, tok, mixed_rows, seed=0)
    print("trained in", round(time.time() - t0), "s | peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
    mix_out = C.generate(mix_trainer.model.eval(), tok, items)
    mix = C.score(items, mix_out, parse_xml_calls)
    print("mixed set + LoRA", json.dumps(C.summary(mix)))
    print("change against the base, 95% CI:", C.paired_bootstrap([r["correct"] for r in base], [r["correct"] for r in mix]))
