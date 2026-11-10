import os, re
import torch
from math_verify import parse, verify

NL = chr(10)
STOPS = [NL + "Question", NL + "[Question]"]       # a base model answers, then starts a new question; stop there
TIMEOUT = None if os.name == "nt" else 5            # math-verify's timeout uses signals, which Windows lacks

def prompt(q):
    return "Question: " + q + NL + "Answer:"

def cut(text):
    # the reply up to the first stop string
    ends = [text.find(s) for s in STOPS if s in text]
    return text[:min(ends)] if ends else text

def final_answer(reply):
    # the text after the last "####" or "answer is", else the last number in the reply
    reply = cut(reply)
    marked = re.findall("(?:####|answer is)([^" + NL + "]*)", reply)
    if marked:
        return marked[-1]
    nums = re.findall("-?[0-9][0-9,]*(?:[.][0-9]+)?", reply)
    return nums[-1] if nums else ""

def correct(reply, gold):
    # 1.0 if math-verify finds the reply's final answer equal to the gold number, else 0.0
    pred = parse(final_answer(reply).replace(",", ""), parsing_timeout=TIMEOUT)
    return float(bool(pred) and verify(parse(gold, parsing_timeout=TIMEOUT), pred, timeout_seconds=TIMEOUT))

def group_advantages(rewards, G, scale=True):
    # GRPO: compare each reply with the other replies to the same prompt; no value model
    r = rewards.view(-1, G)
    adv = r - r.mean(1, keepdim=True)
    if scale:                                        # divide by the group's std (Bessel-corrected, as TRL does)
        adv = adv / (r.std(1, keepdim=True) + 1e-4)
    return adv.reshape(-1)

def rloo_advantages(rewards, G):
    # RLOO: the baseline of each reply is the mean of the other G - 1 replies
    r = rewards.view(-1, G)
    return (r - (r.sum(1, keepdim=True) - r) / (G - 1)).reshape(-1)

gold = "18"
for reply in [" Janet sells 16 - 3 - 4 = 9 eggs. 9 * 2 = 18. The answer is 18." + NL + "Question: What is 2 + 2?",
              " She makes 9 * 2 = $18.00 a day." + NL + NL + "Question: next",
              " 16 - 3 - 4 = 9 eggs, so she makes 9 * 2 = 18 dollars. #### 18",
              " The answer is 16.",
              " She sells the rest at the market."]:
    print(correct(reply, gold), "| final answer", repr(final_answer(reply)), "|", repr(cut(reply)[:60]))

rewards = torch.tensor([1., 0., 0., 1., 0., 0., 0., 1.,     # prompt A: 3 of 8 correct
                        1., 1., 1., 1., 1., 1., 1., 1.,     # prompt B: all correct
                        0., 0., 0., 0., 0., 0., 0., 0.,     # prompt C: none correct
                        0., 0., 0., 0., 0., 0., 0., 1.])    # prompt D: 1 of 8 correct
for name, adv in [("GRPO, scaled by std", group_advantages(rewards, 8)), ("Dr. GRPO, not scaled", group_advantages(rewards, 8, scale=False)),
                  ("RLOO", rloo_advantages(rewards, 8))]:
    a = adv.view(4, 8)
    print(name, "| correct reply in A, D:", round(a[0, 0].item(), 3), round(a[3, 7].item(), 3), "| wrong reply in A, D:",
          round(a[0, 1].item(), 3), round(a[3, 0].item(), 3), "| groups B and C all zero:", bool((a[1:3] == 0).all()))
zero = (rewards.view(-1, 8).std(1) == 0).float().mean().item()
print("share of groups with zero spread, which give no gradient:", zero)
