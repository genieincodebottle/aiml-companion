import glob, time
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

BASE = "Qwen/Qwen3-0.6B-Base"
hub_dir = snapshot_download(HUB, allow_patterns=["lltDPO/sft-s0/*", "lltDomainAdaptation/cpt-r5-lr3e-5/*"])
DIRS = {"base": snapshot_download(BASE), "sft": hub_dir + "/lltDPO/sft-s0", "hindi": hub_dir + "/lltDomainAdaptation/cpt-r5-lr3e-5"}

def load_weights(d):
    # Every tensor of a checkpoint folder, in the dtype it was saved in (bf16 for the base, fp16 for our fine-tunes)
    sd = {}
    for f in sorted(glob.glob(d + "/*.safetensors")):
        sd.update(load_file(f))
    return sd

W = {name: load_weights(d) for name, d in DIRS.items()}
print({name: (len(sd), str(next(iter(sd.values())).dtype)) for name, sd in W.items()})
assert W["base"].keys() == W["sft"].keys() == W["hindi"].keys(), "different tensor names: not the same architecture"

def tau(name, k):
    # Task vector of one tensor: fine-tuned weights minus base weights, in fp32
    return W[name][k].float() - W["base"][k].float()

def group(k):
    if "embed_tokens" in k:
        return "embeddings"
    return "norms" if "norm" in k else ("attention" if "self_attn" in k else "mlp")

stats = {n: {"sq": {}, "nonzero": 0} for n in ("sft", "hindi")}
dot = conflict = both = total = base_sq = 0.0
t0 = time.time()
for k in W["base"]:
    a, b = tau("sft", k), tau("hindi", k)
    for n, d in (("sft", a), ("hindi", b)):
        stats[n]["sq"][group(k)] = stats[n]["sq"].get(group(k), 0.0) + float((d * d).sum())
        stats[n]["nonzero"] += int((d != 0).sum())
    base_sq += float((W["base"][k].float() ** 2).sum())
    dot += float((a * b).sum())
    nz = (a != 0) & (b != 0)
    both += int(nz.sum())
    conflict += int((nz & (a.sign() != b.sign())).sum())
    total += a.numel()

norm = {n: sum(s["sq"].values()) ** 0.5 for n, s in stats.items()}
for n, s in stats.items():
    print(n.ljust(6), "| norm", round(norm[n], 2), "=", round(100 * norm[n] / base_sq ** 0.5, 3), "% of the base norm",
          "| nonzero entries", round(100 * s["nonzero"] / total, 1), "%",
          "| share of the squared norm", {g: round(v / norm[n] ** 2, 3) for g, v in s["sq"].items()})
print("cosine(sft, hindi)", round(dot / (norm["sft"] * norm["hindi"]), 4), "| entries both changed", round(100 * both / total, 1),
      "% | of those, opposite signs", round(100 * conflict / both, 1), "% |", round(time.time() - t0), "s")

k = "model.layers.13.self_attn.q_proj.weight"
for n in ("sft", "hindi"):
    s = torch.linalg.svdvals(tau(n, k))
    print(n.ljust(6), k, "singular values 1, 16, 17 relative to the first:", [round(float(s[i] / s[0]), 4) for i in (0, 15, 16)])
