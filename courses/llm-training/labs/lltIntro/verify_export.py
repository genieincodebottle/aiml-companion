"""Check the browser export of the story model against the PyTorch original.

Loads story-model.json (float16) with the same NumPy forward pass the page runs, and the float32
PyTorch weights (.pt), then compares logits and loss on the held-out story stored in the JSON.
Adds the result to the seed-0 Run Library file under evals.exportCheck.
Usage: python verify_export.py <story-model.json>
"""
import base64
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from runlog import RUNS_DIR
from train_story_model import TinyGPT

path = sys.argv[1]
M = json.load(open(path, encoding="ascii"))
vocab, H, CTX = M["vocab"], M["n_head"], M["ctx"]


def unpack(w):
    return np.frombuffer(base64.b64decode(w["b64"]), np.float16).astype(np.float32).reshape(w["shape"])


def norm(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g


def logits_np(W, ids):
    T = len(ids)
    x = W["tok"][ids] + W["pos"][:T]
    mask = np.triu(np.full((T, T), -1e9), 1)
    for i in range(M["n_layer"]):
        q, k, v = np.split(norm(x, W[f"n1.{i}"]) @ W[f"qkv.{i}"], 3, axis=1)
        q, k, v = (a.reshape(T, H, -1).transpose(1, 0, 2) for a in (q, k, v))
        a = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1]) + mask
        a = np.exp(a - a.max(-1, keepdims=True))
        y = (a / a.sum(-1, keepdims=True)) @ v
        x = x + y.transpose(1, 0, 2).reshape(T, -1) @ W[f"proj.{i}"]
        h = norm(x, W[f"n2.{i}"]) @ W[f"fc.{i}"]
        h = 0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))
        x = x + h @ W[f"out.{i}"]
    return norm(x, W["nf"]) @ W["tok"].T


W16 = {k: unpack(w) for k, w in M["weights"].items()}
text = M["eval_text"][: CTX + 1]
ids = [vocab.index(c) for c in text]
z16 = logits_np(W16, ids[:-1])

model = TinyGPT(len(vocab), M["d_model"], M["n_layer"], H, CTX)
model.load_state_dict(torch.load(os.path.splitext(path)[0] + ".pt"))
model.eval()
with torch.no_grad():
    z32 = model(torch.tensor([ids[:-1]]))[0]
    loss32 = F.cross_entropy(z32, torch.tensor(ids[1:])).item()
loss16 = F.cross_entropy(torch.tensor(z16), torch.tensor(ids[1:])).item()
diff = float(np.abs(z16 - z32.numpy()).max())
res = {"evalChars": len(text), "lossFloat32Torch": round(loss32, 4), "lossFloat16NumPy": round(loss16, 4),
       "maxAbsLogitDiff": round(diff, 4), "jsonBytes": os.path.getsize(path)}
print(res)

run = os.path.join(RUNS_DIR, "lltIntro-t1-story-s0.json")
d = json.load(open(run, encoding="utf-8"))
d["evals"]["exportCheck"] = res
json.dump(d, open(run, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("updated", run)
