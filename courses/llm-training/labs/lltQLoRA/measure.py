"""lltQLoRA CPU measurements (exact figures, no timings). Writes Run Library JSON to runs/lltQLoRA/.

  python measure.py nf4  <weights_dir>   NF4 levels, block-wise absmax, NF4 vs uniform int4 vs FP4 on real
                                         Qwen3.5-9B-Base weights, block size sweep, our NumPy NF4 vs bitsandbytes,
                                         double quantization bytes and error (bitsandbytes CPU backend)
  python measure.py mem  <weights_dir>   parameter counts from the safetensors headers and the predicted NF4 footprint
  python measure.py tryit <weights_dir> <out.json>   the small JSON the Try it widget reads

<weights_dir> is the output of fetch_weights.py. Needs numpy, scipy, torch, bitsandbytes 0.50.2.
"""
import json, os, struct, sys, platform
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "runs", "lltQLoRA")
os.makedirs(RUNS, exist_ok=True)


def env():
    import scipy
    e = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    try:
        import torch, bitsandbytes
        torch.set_num_threads(4)
        e.update(torch=torch.__version__, bitsandbytes=bitsandbytes.__version__)
    except Exception:
        pass
    return e


def nf4_levels(offset=0.9677083):
    # 8 positive quantiles and 7 negative ones of N(0, 1), plus an exact zero, scaled into [-1, 1].
    pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])
    neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])
    v = np.sort(np.concatenate([neg, [0.0], pos]))
    return v / v.max()


INT4 = np.arange(-7, 8) / 7.0                       # symmetric absmax int4: 15 evenly spaced levels
FP4 = np.array(sorted([0.0, 0.0052, 0.6667, 1.0, 0.3333, 0.5, 0.1667, 0.25,
                       -0.0052, -0.6667, -1.0, -0.3333, -0.5, -0.1667, -0.25]))   # bitsandbytes fp4 table


def quantize(w, levels, block=64):
    """Block-wise absmax quantization to a fixed level table. Returns codes, absmax, dequantized."""
    x = w.reshape(-1, block)
    absmax = np.abs(x).max(axis=1, keepdims=True)
    scaled = x / absmax
    mid = (levels[1:] + levels[:-1]) / 2                 # nearest level = bucket between midpoints
    codes = np.searchsorted(mid, scaled)
    deq = levels[codes] * absmax
    return codes, absmax[:, 0], deq.reshape(w.shape)


def rel_err(w, wq):
    w = w.astype(np.float64)
    return float(np.linalg.norm(w - wq.astype(np.float64)) / np.linalg.norm(w))


def load(wdir):
    s = json.load(open(os.path.join(wdir, "weights_summary.json")))
    ws = {n: np.load(os.path.join(wdir, i["file"])) for n, i in s["fetched"].items()}
    return s, ws


def cmd_nf4(wdir):
    import torch
    import bitsandbytes.functional as F
    s, ws = load(wdir)
    lv = nf4_levels()
    bnb_lv = F.get_4bit_type("nf4", device="cpu").numpy()[:16]
    out = {"runId": "lltQLoRA-t1-nf4-v2026-11", "module": "lltQLoRA", "tier": "T1", "label": "measurement:nf4",
           "env": env(), "date": "2026-11-06", "source": s["repo"],
           "nf4Levels": [round(float(v), 6) for v in lv],
           "nf4LevelsMaxAbsDiffVsBitsandbytes": float(np.abs(lv - bnb_lv).max()),
           "offset": 0.9677083, "offsetDerivation": "(1 - 1/(2*15) + 1 - 1/(2*16)) / 2",
           "int4Levels": [round(float(v), 6) for v in INT4], "fp4Levels": FP4.tolist(), "tensors": {}}
    for name, w in ws.items():
        r = {"shape": list(w.shape), "numel": int(w.size), "std": float(w.std()), "absmax": float(np.abs(w).max()),
             "kurtosis": float(((w - w.mean()) ** 4).mean() / w.var() ** 2),
             "blockAbsmaxMedianInStd": float(np.median(np.abs(w.reshape(-1, 64)).max(axis=1)) / w.std()),
             "nf4OffsetInStd": float(norm.ppf(0.9677083))}
        for tag, levels in [("nf4", lv), ("int4", INT4), ("fp4", FP4)]:
            codes, am, deq = quantize(w, levels, 64)
            r[tag] = {"relErr": rel_err(w, deq), "rmse": float(np.sqrt(((w - deq) ** 2).mean())),
                      "levelUse": (np.bincount(codes.ravel(), minlength=len(levels)) / codes.size).round(5).tolist()}
        sweep = {}
        for b in [64, 256, 4096, w.shape[1], w.size]:
            _, _, deq = quantize(w, lv, b)
            sweep[str(b)] = rel_err(w, deq)
        r["nf4BlockSweep"] = sweep
        # Our NumPy NF4 against bitsandbytes on the same tensor (fp32 input, block 64, no double quant).
        t = torch.from_numpy(w)
        q, st = F.quantize_4bit(t, blocksize=64, quant_type="nf4", compress_statistics=False)
        bdeq = F.dequantize_4bit(q, st).numpy()
        packed = q.numpy().ravel()
        bcodes = np.empty(packed.size * 2, dtype=np.int64)
        bcodes[0::2] = packed >> 4
        bcodes[1::2] = packed & 15
        ours, _, odeq = quantize(w, lv, 64)
        r["vsBitsandbytes"] = {"codesEqualFraction": float((bcodes[:w.size] == ours.ravel()).mean()),
                               "codesDiffer": int((bcodes[:w.size] != ours.ravel()).sum()),
                               "dequantMaxAbsDiff": float(np.abs(bdeq - odeq).max()),
                               "bnbRelErr": rel_err(w, bdeq)}
        # Double quantization: the fp32 absmax constants are themselves quantized to 8 bits in blocks of 256.
        q2, st2 = F.quantize_4bit(t, blocksize=64, quant_type="nf4", compress_statistics=True)
        d2 = F.dequantize_4bit(q2, st2).numpy()
        bytes_single = q.numel() * q.element_size() + st.absmax.numel() * st.absmax.element_size()
        bytes_dq = (q2.numel() * q2.element_size() + st2.absmax.numel() * st2.absmax.element_size()
                    + st2.state2.absmax.numel() * st2.state2.absmax.element_size() + 4)
        r["doubleQuant"] = {"absmaxDtype": str(st2.absmax.dtype), "absmaxCount": int(st2.absmax.numel()),
                            "state2Blocksize": int(st2.state2.blocksize), "state2AbsmaxCount": int(st2.state2.absmax.numel()),
                            "bitsPerParamSingle": bytes_single * 8 / w.size, "bitsPerParamDouble": bytes_dq * 8 / w.size,
                            "relErrSingle": rel_err(w, bdeq), "relErrDouble": rel_err(w, d2),
                            "dequantMaxAbsDiffSingleVsDouble": float(np.abs(bdeq - d2).max())}
        out["tensors"][name] = r
        print(name, {k: (r[k]["relErr"] if isinstance(r[k], dict) and "relErr" in r[k] else None) for k in ("nf4", "int4", "fp4")},
              "sweep", {k: round(v, 5) for k, v in sweep.items()}, "bnb", r["vsBitsandbytes"], "dq", r["doubleQuant"])
    json.dump(out, open(os.path.join(RUNS, out["runId"] + ".json"), "w"), indent=1)
    print("wrote", out["runId"])


def headers():
    import requests
    base = "https://huggingface.co/Qwen/Qwen3.5-9B-Base/resolve/main/"
    idx = requests.get(base + "model.safetensors.index.json", timeout=60).json()
    allh = {}
    for sh in sorted(set(idx["weight_map"].values())):
        r = requests.get(base + sh, headers={"Range": "bytes=0-7"}, timeout=60)
        n = struct.unpack("<Q", r.content)[0]
        h = json.loads(requests.get(base + sh, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=60).content)
        allh.update({k: v for k, v in h.items() if k != "__metadata__"})
    return allh


def cmd_mem(wdir):
    h = headers()
    text = {k: v for k, v in h.items() if k.startswith("model.language_model.") or k == "lm_head.weight"}
    # Every 2-D weight inside the decoder layers is an nn.Linear, which bitsandbytes replaces with Linear4bit.
    lin = {k: v for k, v in text.items() if k.startswith("model.language_model.layers.") and len(v["shape"]) == 2}
    L = sum(int(np.prod(v["shape"])) for v in lin.values())
    T = sum(int(np.prod(v["shape"])) for v in text.values())
    other = T - L
    kinds = {}
    for k, v in lin.items():
        kk = k.split(".")[-2]
        kinds[kk] = kinds.get(kk, 0) + int(np.prod(v["shape"]))
    nf4_bytes_single = L * (0.5 + 4 / 64)
    nf4_bytes_dq = L * (0.5 + 1 / 64 + 4 / (64 * 256))
    out = {"runId": "lltQLoRA-t1-memory-v2026-11", "module": "lltQLoRA", "tier": "T1", "label": "measurement:memory",
           "env": env(), "date": "2026-11-06", "source": "safetensors headers of Qwen/Qwen3.5-9B-Base (HTTP range reads)",
           "textParams": T, "quantizedLinearParams": L, "nonQuantizedTextParams": other,
           "linearParamsByName": kinds, "linearTensorCount": len(lin),
           "bf16TextGB": T * 2 / 1e9,
           "predictedNF4GB_singleQuant": (nf4_bytes_single + other * 2) / 1e9,
           "predictedNF4GB_doubleQuant": (nf4_bytes_dq + other * 2) / 1e9,
           "predictedNF4GB_doubleQuant_linearOnly": nf4_bytes_dq / 1e9,
           "nonQuantizedFp16GB": other * 2 / 1e9,
           "numelCountedByTorchAfter4bit": L // 2 + other}
    json.dump(out, open(os.path.join(RUNS, out["runId"] + ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_tryit(wdir, outp):
    s, ws = load(wdir)
    name = "model.language_model.layers.14.mlp.down_proj.weight"
    w = ws[name]
    lv = nf4_levels()
    x = w.reshape(-1, 64)
    scaled = (x / np.abs(x).max(axis=1, keepdims=True)).ravel()
    edges = np.linspace(-1, 1, 81)
    hist_scaled = np.histogram(scaled, bins=edges)[0]
    sd = float(w.std())
    redges = np.linspace(-6 * sd, 6 * sd, 81)
    hist_raw = np.histogram(w.ravel(), bins=redges)[0]
    schemes = {}
    for tag, levels in [("nf4", lv), ("int4", INT4), ("fp4", FP4)]:
        codes, _, deq = quantize(w, levels, 64)
        schemes[tag] = {"levels": [round(float(v), 5) for v in levels], "relErr": round(rel_err(w, deq), 5),
                        "levelUse": (np.bincount(codes.ravel(), minlength=len(levels)) / codes.size).round(4).tolist()}
    blk_sweep = {}
    for b in [64, 256, 4096, w.shape[1]]:
        _, _, d = quantize(w, lv, b)
        _, _, d2 = quantize(w, INT4, b)
        blk_sweep[str(b)] = {"nf4": round(rel_err(w, d), 5), "int4": round(rel_err(w, d2), 5)}
    # One real block of 64 weights so the widget can show quantization of a single block.
    rng = np.random.default_rng(0)
    bi = int(rng.integers(0, x.shape[0]))
    block = x[bi]
    data = {"source": s["repo"], "tensor": name.replace("model.language_model.", ""), "shape": list(w.shape),
            "numel": int(w.size), "std": round(sd, 6), "absmax": round(float(np.abs(w).max()), 5),
            "runId": "lltQLoRA-t1-nf4-v2026-11", "blocksize": 64,
            "histScaled": {"edges": [round(float(e), 3) for e in edges], "counts": hist_scaled.tolist()},
            "histRaw": {"edges": [round(float(e), 6) for e in redges], "counts": hist_raw.tolist(),
                        "outsideRange": int((np.abs(w) > 6 * sd).sum())},
            "schemes": schemes, "blockSweep": blk_sweep,
            "exampleBlock": {"index": bi, "values": [round(float(v), 6) for v in block]}}
    json.dump(data, open(outp, "w"), separators=(",", ":"))
    print("tryit json", os.path.getsize(outp), "bytes", {k: v["relErr"] for k, v in schemes.items()}, blk_sweep)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "nf4":
        cmd_nf4(sys.argv[2])
    elif c == "mem":
        cmd_mem(sys.argv[2])
    elif c == "tryit":
        cmd_tryit(sys.argv[2], sys.argv[3])
