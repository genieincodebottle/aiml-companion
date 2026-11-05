import numpy as np

@torch.no_grad()
def logprobs(model, ctx, resp):
    # Row t holds log-probabilities, the natural log of each probability, for response token t over the whole vocabulary.
    x = torch.tensor([ctx + resp])
    logits = model(input_ids=x).logits[0, len(ctx) - 1 : len(ctx) - 1 + len(resp)]
    return torch.log_softmax(logits.double(), dim=-1)

def kl_and_rank(lp_i, lp_b, resp):
    kl = (lp_i.exp() * (lp_i - lp_b)).sum(-1)          # KL(instruct || base) per token, in nats
    chosen = lp_b[torch.arange(len(resp)), torch.tensor(resp)]
    rank = (lp_b > chosen[:, None]).sum(-1) + 1         # 1 = the base model also put this token first
    return kl, rank

ctx = chat("Why is the sky blue?")
resp = greedy(inst, tok_i, ctx, 128)
kl, rank = kl_and_rank(logprobs(inst, ctx, resp), logprobs(base, ctx, resp), resp)
for t in range(12):
    print(str(t + 1).rjust(3), repr(tok_i.decode([resp[t]])).ljust(16), "KL", round(float(kl[t]), 3), "base rank", int(rank[t]))
print(len(resp), "tokens, mean KL", round(float(kl.mean()), 3), "nats")
