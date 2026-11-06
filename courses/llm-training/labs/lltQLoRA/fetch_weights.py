"""Read real Qwen3.5-9B-Base weights without downloading the 19 GB checkpoint.

A safetensors file starts with an 8-byte header length and a JSON header that gives every tensor's
dtype, shape and byte range. HTTP range requests read the four headers (to count parameters) and a
few whole tensors (for the NF4 measurements). Writes bf16 tensors as float32 .npy files and a JSON
summary of the parameter counts.

Usage: python fetch_weights.py <out_dir>
"""
import json, os, struct, sys
import numpy as np
import requests

REPO = "Qwen/Qwen3.5-9B-Base"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

WANT = [
    "model.language_model.layers.14.mlp.down_proj.weight",
    "model.language_model.layers.14.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.15.self_attn.q_proj.weight",
]


def get_range(url, start, end):
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600, allow_redirects=True)
    r.raise_for_status()
    assert len(r.content) == end - start + 1, (len(r.content), end - start + 1)
    return r.content


index = requests.get(BASE + "model.safetensors.index.json", timeout=60).json()
shards = sorted(set(index["weight_map"].values()))
headers = {}
for s in shards:
    n = struct.unpack("<Q", get_range(BASE + s, 0, 7))[0]
    headers[s] = (n, json.loads(get_range(BASE + s, 8, 8 + n - 1)))

counts = {"text": 0, "vision": 0, "mtp": 0, "other": 0}
dtypes = {}
for s, (n, h) in headers.items():
    for name, info in h.items():
        if name == "__metadata__":
            continue
        k = int(np.prod(info["shape"]))
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + k
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            counts["text"] += k
        elif name.startswith("model.visual."):
            counts["vision"] += k
        elif name.startswith("mtp."):
            counts["mtp"] += k
        else:
            counts["other"] += k

h_all = {name: (s, n, info) for s, (n, h) in headers.items() for name, info in h.items() if name != "__metadata__"}
emb = h_all["model.language_model.embed_tokens.weight"][2]["shape"]
head = h_all["lm_head.weight"][2]["shape"]
fetched = {}
for name in WANT:
    s, n, info = h_all[name]
    a, b = info["data_offsets"]
    raw = get_range(BASE + s, 8 + n + a, 8 + n + b - 1)
    assert info["dtype"] == "BF16"
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16   # bf16 -> fp32 is a 16-bit left shift
    w = u16.view(np.float32).reshape(info["shape"])
    fn = name.replace("model.language_model.", "").replace(".", "_") + ".npy"
    np.save(os.path.join(OUT, fn), w)
    fetched[name] = {"file": fn, "shape": info["shape"], "shard": s, "bytes": b - a,
                     "mean": float(w.mean()), "std": float(w.std()), "absmax": float(np.abs(w).max())}
    print(name, info["shape"], round(float(w.std()), 6))

summary = {"repo": REPO, "shards": shards, "paramCounts": counts, "dtypeCounts": dtypes,
           "embedShape": emb, "lmHeadShape": head, "fetched": fetched}
json.dump(summary, open(os.path.join(OUT, "weights_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "fetched"}, indent=1))
