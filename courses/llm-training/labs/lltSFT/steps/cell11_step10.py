def cross_doc_collator(collate, restart_positions, explicit_mask=True):
    # Same packed rows and labels as TRL's collator, but every token may attend to every earlier token in the row.
    # explicit_mask=True passes a full causal mask, so SDPA gets an explicit mask, as it does with the document mask.
    def wrapped(examples):
        b = collate(examples)
        n = b["input_ids"].shape[1]
        if not restart_positions:
            b["position_ids"] = torch.arange(n)[None]
        if explicit_mask:
            b["attention_mask"] = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
        for k in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx"):
            b.pop(k, None)
        return b
    return wrapped

if RUN_EXPERIMENT:
    for name, restart, explicit in [("naive packing", False, False),
                                    ("cross-document, positions restart", True, True),
                                    ("cross-document, positions run on", False, True)]:
        t = fresh_trainer()
        t.data_collator = cross_doc_collator(t.data_collator, restart, explicit)
        train_and_score(t, name)
        del t
    train_and_score(fresh_trainer(packing=False, per_device_train_batch_size=4), "padded")
