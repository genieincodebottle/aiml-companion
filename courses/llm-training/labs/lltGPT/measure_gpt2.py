"""Step 7 measurements: our gpt2-config build against Hugging Face GPT-2 124M in float32.

Records the logit match, the gap between HF's own two attention paths and a float64 run, and what
each common loading mistake costs. The square attn.c_proj mistake is the Diagnose it run.
Writes runs/lltGPT/lltGPT-t1-gpt2-match.json and lltGPT-t1-broken-conv1d-square.json.
"""
import contextlib
import io
import time
from pathlib import Path

from runlog import write_run

HERE = Path(__file__).resolve().parent
ns = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for name in ["step2", "step3", "step4"]:
        exec(compile((HERE / "steps" / f"{name}.py").read_text(encoding="utf-8"), name, "exec"), ns)
src = (HERE / "steps" / "step7.py").read_text(encoding="utf-8")
head = src.split("ours = GPT(GPT2)")[0]
exec(compile(head, "step7-head", "exec"), ns)
torch, F, nn, replace = ns["torch"], ns["F"], ns["nn"], ns["replace"]
torch.set_num_threads(4)
GPT, GPT2, gsd, hf, tok, CONV1D, RENAME = (ns[k] for k in ["GPT", "GPT2", "gpt2_state_dict", "hf", "tok", "CONV1D", "RENAME"])
from transformers import GPT2LMHeadModel  # noqa: E402

TEXT = "Once upon a time, a small robot lived in a quiet town. Every morning it walked to the park and counted the birds."
PROMPT = "Once upon a time, a small robot"
assert TEXT in src and PROMPT in src, "step7 text changed; update measure_gpt2.py"
ids = tok(TEXT, return_tensors="pt").input_ids
pids = tok(PROMPT, return_tensors="pt").input_ids
sd = hf.state_dict()


def run(model, x=ids):
    with torch.no_grad():
        return model(x) if not hasattr(model, "transformer") else model(x).logits


def greedy(model, n=12):
    p = pids.clone()
    with torch.no_grad():
        for _ in range(n):
            p = torch.cat([p, run(model, p)[:, -1:].argmax(-1)], dim=1)
    return tok.decode(p[0])


def build(cfg=GPT2, state=None):
    m = GPT(cfg).eval()
    m.load_state_dict(state if state is not None else gsd(sd))
    return m


t = time.time()
ref = run(hf)
ours = build()
out = run(ours)
eager = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32, attn_implementation="eager").eval()
ref_eager = run(eager)
o64 = GPT(GPT2).double().eval()
o64.load_state_dict({k: v.double() for k, v in gsd(sd).items()})
r64 = run(o64).float()
loss = lambda z: F.cross_entropy(z[0, :-1], ids[0, 1:]).item()  # noqa: E731
match = {
    "tokens": ids.shape[1], "largestAbsLogit": round(ref.abs().max().item(), 2),
    "oursVsHfSdpa": (out - ref).abs().max().item(), "oursVsHfEager": (out - ref_eager).abs().max().item(),
    "hfSdpaVsHfEager": (ref - ref_eager).abs().max().item(), "oursFp32VsOursFp64": (out - r64).abs().max().item(),
    "hfSdpaVsOursFp64": (ref - r64).abs().max().item(),
    "maxRelDiff": ((out - ref).abs() / ref.abs().clamp_min(1)).max().item(),
    "sameArgmax": bool((out.argmax(-1) == ref.argmax(-1)).all()),
    "allcloseRtol1e-5Atol1e-4": bool(torch.allclose(out, ref, rtol=1e-5, atol=1e-4)),
    "lossOurs": round(loss(out), 4), "lossHf": round(loss(ref), 4),
    "greedyOurs": greedy(ours), "greedyHf": tok.decode(hf.generate(pids, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)[0]),
    "paramsOurs": sum(p.numel() for p in ours.parameters()), "paramsHf": sum(p.numel() for p in hf.parameters()),
}
print(match, flush=True)

# What each mistake costs, one at a time
mistakes = {}


def rec(name, model):
    z = run(model)
    mistakes[name] = {"maxAbsDiff": round((z - ref).abs().max().item(), 4), "loss": round(loss(z), 4),
                      "sameArgmaxFrac": round((z.argmax(-1) == ref.argmax(-1)).float().mean().item(), 3),
                      "greedy": greedy(model)}
    print(name, mistakes[name], flush=True)


m = build(); [setattr(b.mlp, "forward", (lambda self, x: self.down(F.gelu(self.up(x)))).__get__(b.mlp)) for b in m.blocks]
rec("exact GELU instead of tanh", m)
rec("LayerNorm eps 1e-6 instead of 1e-5", build(replace(GPT2, eps=1e-6)))
# The shape-check loader: transpose only where the shapes disagree. attn.c_proj is 768 x 768.
broken = gsd(sd)
for key, w in sd.items():
    if key.startswith("transformer.h.") and key.endswith("attn.c_proj.weight"):
        layer = key.split(".")[2]
        broken[f"blocks.{layer}.attn.proj.weight"] = w   # left in (in, out) layout
rec("square attn.c_proj left untransposed", build(state=broken))
nof = gsd(sd)
nof["norm.weight"] = torch.ones(768); nof["norm.bias"] = torch.zeros(768)
rec("final LayerNorm weights not loaded", build(state=nof))
# No transposes at all: strict loading raises
raw = gsd(sd)
for key, w in sd.items():
    if key.startswith("transformer.h.") and key.endswith(".weight") and w.dim() == 2:
        layer, rest = key[len("transformer.h."):].split(".", 1)
        module = rest.rsplit(".", 1)[0]
        raw[f"blocks.{layer}.{RENAME[module]}.weight"] = w
try:
    GPT(GPT2).load_state_dict(raw)
    no_transpose_error = None
except RuntimeError as e:
    no_transpose_error = str(e).splitlines()[:3]
print("no transpose:", no_transpose_error, flush=True)
sec = time.time() - t

cfg = {"model": "openai-community/gpt2", "dtype": "float32", "threads": 4, "text": TEXT, "prompt": PROMPT}
write_run("lltGPT-t1-gpt2-match", "T1", "baseline", cfg, None, [], evals={"match": match, "mistakes": mistakes, "noTransposeError": no_transpose_error}, wall_clock_sec=round(sec, 1))
write_run("lltGPT-t1-broken-conv1d-square", "T1", "broken:conv1d-square-untransposed", cfg, None, [],
          evals={"broken": mistakes["square attn.c_proj left untransposed"], "fixed": {"maxAbsDiff": match["oursVsHfSdpa"], "loss": match["lossOurs"], "greedy": match["greedyOurs"]}},
          wall_clock_sec=None)
