import subprocess, sys, tarfile, urllib.request
from gguf import GGUFReader
from gguf.quants import dequantize

TAG = "b12182"                                   # llama.cpp release tag; quote the tag you used
REL = "https://github.com/ggml-org/llama.cpp/releases/download/" + TAG + "/"
BIN = os.path.abspath("llama-" + TAG)            # prebuilt CPU binaries for Linux x64
SRC = os.path.abspath("llama.cpp-" + TAG)        # the same tag's source, for convert_hf_to_gguf.py and gguf-py
if not os.path.isdir(BIN):
    urllib.request.urlretrieve(REL + "llama-" + TAG + "-bin-ubuntu-x64.tar.gz", "bin.tgz")
    tarfile.open("bin.tgz").extractall(".", filter="data")
if not os.path.isdir(SRC):
    urllib.request.urlretrieve("https://github.com/ggml-org/llama.cpp/archive/refs/tags/" + TAG + ".tar.gz", "src.tgz")
    tarfile.open("src.tgz").extractall(".", filter="data")

def sh(cmd, **kw):
    # Run a command, return seconds taken; raise with its last output lines if it fails
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return round(time.time() - t, 1), p.stdout + p.stderr

env = dict(os.environ, PYTHONPATH=SRC + "/gguf-py")
sec, _ = sh([sys.executable, SRC + "/convert_hf_to_gguf.py", "merged", "--outtype", "f16", "--outfile", "merged-f16.gguf"], env=env)
gguf = {"f16": {"file": "merged-f16.gguf", "sec": sec}}
for q in ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]:
    sec, _ = sh([BIN + "/llama-quantize", "merged-f16.gguf", "merged-" + q + ".gguf", q])
    gguf[q] = {"file": "merged-" + q + ".gguf", "sec": sec}
for q, g in gguf.items():
    g["MiB"] = round(os.path.getsize(g["file"]) / 2 ** 20, 1)
    print(q.ljust(7), g["MiB"], "MiB | made in", g["sec"], "s")

# Read the quantized weights back into the PyTorch model, so each GGUF file gets the same paired panel as the merges
NAMES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
         "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "attn_norm": "input_layernorm",
         "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

def gguf_weights(path):
    sd, kinds = {}, {}
    for t in GGUFReader(path).tensors:
        w = torch.from_numpy(dequantize(t.data, t.tensor_type).copy()).reshape([int(x) for x in reversed(t.shape)])
        part = t.name.split(".")
        if t.name == "token_embd.weight":
            k = "model.embed_tokens.weight"
        elif t.name == "output_norm.weight":
            k = "model.norm.weight"
        else:
            k = "model.layers." + part[1] + "." + NAMES[part[2]] + ".weight"
        sd[k] = w
        kinds[t.tensor_type.name] = kinds.get(t.tensor_type.name, 0) + 1
    return sd, kinds

merged_sd = {k: v.float().cpu() for k, v in model.state_dict().items() if k != "lm_head.weight"}
for q in ["f16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"] if DEV.type == "cuda" else ["Q4_K_M"]:
    sd, kinds = gguf_weights(gguf[q]["file"])
    gap = max(float((sd[k] - merged_sd[k]).abs().max()) for k in merged_sd)
    load_into(sd)
    runs["gguf " + q] = panel(model)
    runs["gguf " + q]["kinds"], runs["gguf " + q]["MiB"] = kinds, gguf[q]["MiB"]
    show("GGUF " + q, runs["gguf " + q])
    print("   tensor types", kinds, "| largest weight change", round(gap, 4),
          "| Hindi bpb vs merged", ci(runs[best], runs["gguf " + q], "hi"), "| reply NLL vs merged", ci(runs[best], runs["gguf " + q], "reply"))
load_into(merged_sd)

# llama.cpp's own check: perplexity on the same Hindi texts, with llama.cpp's CPU kernels
open("hindi.txt", "w", encoding="utf-8").write((NL + NL).join(hindi))
for q in ["f16", "Q4_K_M", "Q2_K"]:
    sec, out = sh([BIN + "/llama-perplexity", "-m", gguf[q]["file"], "-f", "hindi.txt", "-c", "512", "--chunks", "40", "-t", str(os.cpu_count())])
    gguf[q]["ppl"] = [l for l in out.splitlines() if "Final estimate" in l][0].split("PPL = ")[1].strip()
    print(q.ljust(7), "llama-perplexity on the Hindi texts, 40 chunks of 512:", gguf[q]["ppl"], "|", sec, "s")
runs["gguf"] = gguf
