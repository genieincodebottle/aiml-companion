"""Check the step 7 tolerance claim: allclose(rtol=1e-5, atol=1e-4) passes the correct builds and
fails the two small mistakes. Appends the result to runs/lltGPT/lltGPT-t1-gpt2-match.json (evals.tolerance)."""
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
import torch  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ns = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for name in ["step2", "step3", "step4"]:
        exec(compile((HERE / "steps" / f"{name}.py").read_text(encoding="utf-8"), name, "exec"), ns)
    exec(compile((HERE / "steps" / "step7.py").read_text(encoding="utf-8").split("ours = GPT(GPT2)")[0], "s7", "exec"), ns)
GPT, GPT2, gsd, hf, tok, F, replace = (ns[k] for k in ["GPT", "GPT2", "gpt2_state_dict", "hf", "tok", "F", "replace"])
from transformers import GPT2LMHeadModel  # noqa: E402

TEXT = "Once upon a time, a small robot lived in a quiet town. Every morning it walked to the park and counted the birds."
ids = tok(TEXT, return_tensors="pt").input_ids
sd = hf.state_dict()
ok = lambda a, b: bool(torch.allclose(a, b, rtol=1e-5, atol=1e-4))  # noqa: E731
with torch.no_grad():
    ref = hf(ids).logits
    eager = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", dtype=torch.float32, attn_implementation="eager").eval()
    ref_e = eager(ids).logits
    ours = GPT(GPT2).eval(); ours.load_state_dict(gsd(sd)); out = ours(ids)
    g = GPT(GPT2).eval(); g.load_state_dict(gsd(sd))
    for b in g.blocks:
        b.mlp.forward = (lambda self, x: self.down(F.gelu(self.up(x)))).__get__(b.mlp)
    e = GPT(replace(GPT2, eps=1e-6)).eval(); e.load_state_dict(gsd(sd))
    res = {"oursVsHfSdpa": ok(out, ref), "oursVsHfEager": ok(out, ref_e), "hfEagerVsHfSdpa": ok(ref_e, ref),
           "exactGelu": ok(g(ids), ref), "eps1e-6": ok(e(ids), ref),
           "absOnly1e-4_hfEagerVsSdpa": bool(torch.allclose(ref_e, ref, rtol=0, atol=1e-4))}
print(res)
p = HERE.parent.parent / "runs" / "lltGPT" / ("lltGPT-t1-gpt2-match" + os.environ.get("LLT_RUN_SUFFIX", "") + ".json")
d = json.loads(p.read_text(encoding="utf-8"))
d["evals"]["tolerance"] = {"check": "torch.allclose(a, b, rtol=1e-5, atol=1e-4)", **res}
p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
print("updated", p)
