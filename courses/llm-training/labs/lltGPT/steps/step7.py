from transformers import AutoTokenizer, GPT2LMHeadModel

hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

# HF GPT-2 keeps these four layers as Conv1D, weight shape (in, out). nn.Linear wants (out, in).
CONV1D = {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}
RENAME = {"ln_1": "n1", "ln_2": "n2", "attn.c_attn": "attn.qkv", "attn.c_proj": "attn.proj",
          "mlp.c_fc": "mlp.up", "mlp.c_proj": "mlp.down"}

def gpt2_state_dict(sd):
    out = {"tok.weight": sd["transformer.wte.weight"], "pos.weight": sd["transformer.wpe.weight"],
           "norm.weight": sd["transformer.ln_f.weight"], "norm.bias": sd["transformer.ln_f.bias"]}
    out["head.weight"] = out["tok.weight"]
    for key, w in sd.items():
        if key.startswith("transformer.h."):
            layer, rest = key[len("transformer.h."):].split(".", 1)
            module, kind = rest.rsplit(".", 1)
            if module in CONV1D and kind == "weight":
                w = w.t()
            out[f"blocks.{layer}.{RENAME[module]}.{kind}"] = w
    return out

ours = GPT(GPT2).eval()
ours.load_state_dict(gpt2_state_dict(hf.state_dict()))   # strict, so a missing or extra key raises
text = "Once upon a time, a small robot lived in a quiet town. Every morning it walked to the park and counted the birds."
ids = tok(text, return_tensors="pt").input_ids
with torch.no_grad():
    ref, out = hf(ids).logits, ours(ids)
diff = (out - ref).abs().max().item()
print(f"{ids.shape[1]} tokens  largest |logit| {ref.abs().max():.1f}  max abs difference {diff:.2e}"
      f"  same top token everywhere {bool((out.argmax(-1) == ref.argmax(-1)).all())}")
print(f"loss  ours {F.cross_entropy(out[0, :-1], ids[0, 1:]):.4f}  HF {F.cross_entropy(ref[0, :-1], ids[0, 1:]):.4f}")
assert torch.allclose(out, ref, rtol=1e-5, atol=1e-4), "logits differ; check the Conv1D transposes and the GELU variant"

prompt = tok("Once upon a time, a small robot", return_tensors="pt").input_ids
with torch.no_grad():
    for _ in range(12):
        prompt = torch.cat([prompt, ours(prompt)[:, -1:].argmax(-1)], dim=1)
print(repr(tok.decode(prompt[0])))
