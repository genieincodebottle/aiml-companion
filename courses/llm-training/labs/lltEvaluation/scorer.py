"""Our own multiple-choice scorer, written to match lm-evaluation-harness 0.4.13 (HFLM).

loglik() reproduces HFLM._encode_pair: encode context + continuation together, take the
continuation tokens as everything after the context's tokens, then sum the log-probabilities
of those tokens. acc picks the highest sum. acc_norm divides each sum by the character length
of the choice text, as lm-eval's multiple_choice process_results does.
"""
import re
import numpy as np
import torch


def encode_pair(tok, context, continuation):
    # lm-eval moves trailing spaces of the context onto the continuation.
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


@torch.no_grad()
def logliks(model, tok, context, continuations):
    """Summed log-probability of each continuation given the context (one padded batch)."""
    pairs = [encode_pair(tok, context, c) for c in continuations]
    seqs = [c + k for c, k in pairs]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
    out = []
    for i, (c, k) in enumerate(pairs):
        pos = torch.arange(len(c) - 1, len(c) + len(k) - 1)
        out.append(float(logp[i, pos, torch.tensor(k)].sum()))
    return np.array(out)


def score_item(lls, choice_texts, gold):
    lens = np.array([float(len(t)) for t in choice_texts])
    return int(np.argmax(lls) == gold), int(np.argmax(lls / lens) == gold)


LETTERS = "ABCDE"


def arc_gold(doc):
    return doc["choices"]["label"].index(doc["answerKey"])


def arc_formats(doc):
    """Five prompt formats for one ARC item: (context, continuations, choice texts for acc_norm)."""
    q, ch = doc["question"], doc["choices"]["text"]
    lettered = "".join(f"\n{LETTERS[i]}. {c}" for i, c in enumerate(ch))
    paren = "".join(f"\n({LETTERS[i].lower()}) {c}" for i, c in enumerate(ch))
    let = [LETTERS[i] for i in range(len(ch))]
    return {
        "cloze_question_answer": (f"Question: {q}\nAnswer:", [" " + c for c in ch], ch),
        "cloze_q_a": (f"Q: {q}\nA:", [" " + c for c in ch], ch),
        "cloze_bare": (q, [" " + c for c in ch], ch),
        "letters_mmlu_style": (f"Question: {q}{lettered}\nAnswer:", [" " + x for x in let], let),
        "letters_paren_lower": (f"{q}{paren}\nThe answer is (", [x.lower() for x in let], let),
    }


def parse_letter(text, n):
    m = re.search(r"\b([A-" + LETTERS[n - 1] + r"])\b", text.strip())
    return LETTERS.index(m.group(1)) if m else -1
