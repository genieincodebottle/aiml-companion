"""Reference mismatch, measured on a CPU: the policy starts from our SFT model, the reference is the base model.

  python measure_refmismatch.py <path to sft-model>
Scores the first 32 held-out pairs (the page's selection) with the SFT model and with Qwen3-0.6B-Base, and reports the
DPO loss and implicit-reward accuracy at step 0, before any training, under the right and the wrong reference.
Writes runs/lltDPO/lltDPO-t1-refmismatch.json.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlog import write_run  # noqa: E402

torch.set_num_threads(4)
SFT = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
NL = chr(10)
TEMPLATE = "### Instruction:" + NL + "{}" + NL + NL + "### Response:" + NL


def encode(prompt, reply):
    p = tok(TEMPLATE.format(prompt)).input_ids
    return p, tok(TEMPLATE.format(prompt) + reply).input_ids + [tok.eos_token_id]


held = []
for ex in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs"):
    c, r = ex["chosen"][-1]["content"].strip(), ex["rejected"][-1]["content"].strip()
    if c == r:
        continue
    p, ci = encode(ex["prompt"], c)
    ri = encode(ex["prompt"], r)[1]
    if len(ci) <= 512 and len(ri) <= 512:
        held.append((p, ci, ri))
    if len(held) == 32:
        break


def logps(model, rows, plen):
    out = []
    for r in rows:
        ids = torch.tensor([r])
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None]).squeeze(-1)
        out.append(float(lp[plen[len(out)] - 1:].sum()))
    return np.array(out)


t0 = time.time()
rows = [h[1] for h in held] + [h[2] for h in held]
plen = [len(h[0]) for h in held] * 2
res = {}
for name, path, dtype in [("sft", SFT, torch.float32), ("base", "Qwen/Qwen3-0.6B-Base", torch.float32)]:
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
    res[name] = logps(m, rows, plen)
    del m
n = len(held)
out = {}
for ref in ("sft", "base"):
    dc = res["sft"][:n] - res[ref][:n]
    dr = res["sft"][n:] - res[ref][n:]
    z = 0.1 * (dc - dr)
    out[ref] = {"lossStep0": round(float(np.mean(np.logaddexp(0, -z))), 4), "accuracyStep0": round(float((dc > dr).mean()), 3),
                "meanChosenLogRatio": round(float(dc.mean()), 2), "meanRejectedLogRatio": round(float(dr.mean()), 2),
                "maxAbsLogRatio": round(float(np.abs(np.concatenate([dc, dr])).max()), 2)}
print(out)
write_run("lltDPO-t1-refmismatch", "T1", "broken:reference-mismatch",
          {"policy": "lltDPO sft-s0 (the SFT model)", "references": ["sft-s0 (right)", "Qwen/Qwen3-0.6B-Base (wrong)"], "pairs": n,
           "beta": 0.1, "selection": "first 32 held-out test_prefs pairs that fit 512 tokens, file order"},
          0, [], evals=out, wall_clock_sec=round(time.time() - t0, 1))
