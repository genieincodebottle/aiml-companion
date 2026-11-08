def build_labels(ids, header, im_end, newline):
    # Loss on each assistant turn: every token after the header up to and including <|im_end|> and the newline
    # after it, which is the span TRL's training template marks with {% generation %}. Everything else is -100:
    # the tool prompt, the user turns and every tool response.
    labels = [-100] * len(ids)
    h = len(header)
    t = 0
    while t < len(ids):
        if ids[t:t + h] == header:
            t += h
            while t < len(ids):
                labels[t] = ids[t]
                t += 1
                if ids[t - 1] == im_end:
                    if t < len(ids) and ids[t] == newline:
                        labels[t] = ids[t]
                        t += 1
                    break
        else:
            t += 1
    return labels

HEADER = tok.encode("<|im_start|>assistant" + C.NL, add_special_tokens=False)
IM_END, NEWLINE = tok.convert_tokens_to_ids("<|im_end|>"), tok.encode(C.NL, add_special_tokens=False)[0]
ids = render(train[N_PER_SOURCE])                  # the first glaive chat: a call, a tool response, then an answer
labels = build_labels(ids, HEADER, IM_END, NEWLINE)
shown = "".join("[" + tok.decode([i]) + "]" if l != -100 else tok.decode([i]) for i, l in zip(ids, labels))
print(shown[shown.index("<|im_start|>user"):][:900])

rows = []
for c in train:
    ids = render(c)
    rows.append({"input_ids": ids, "labels": build_labels(ids, HEADER, IM_END, NEWLINE)})
for name, part in [("singleturn", rows[:N_PER_SOURCE]), ("glaive", rows[N_PER_SOURCE:])]:
    n_tok = sum(len(r["input_ids"]) for r in part)
    n_loss = sum(sum(l != -100 for l in r["labels"]) for r in part)
    print(name, "| tokens", n_tok, "| share with loss", round(n_loss / n_tok, 3))
