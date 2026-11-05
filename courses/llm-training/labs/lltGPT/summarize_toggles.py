"""Print the step 8 table and the broken-run curves from the Run Library files."""
import json
import sys
from pathlib import Path

TIER = sys.argv[1] if len(sys.argv) > 1 else "t2"   # t2 = Kaggle T4 fp32 runs, t1 = CPU

RUNS = Path(__file__).resolve().parent.parent.parent / "runs" / "lltGPT"
NAMES = [("modern", "Modern"), ("learned-pos", "Learned positions"), ("gelu", "GELU MLP"),
         ("layernorm", "LayerNorm"), ("postnorm", "Post-norm"), ("gpt2", "gpt2 config")]
for key, label in NAMES:
    vals, params, curves = [], None, []
    for s in (0, 1):
        p = RUNS / f"lltGPT-{TIER}-toggle-{key}-s{s}.json"
        if not p.exists():
            vals.append(None)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        vals.append(d["evals"]["finalValLoss"])
        params = d["config"]["params"]
        curves.append([m["valLoss"] for m in d["metrics"]])
    mean = sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals))
    if params is None:
        continue
    print(f"{label:20} {params:>10,}  s0 {vals[0]}  s1 {vals[1]}  mean {mean:.4f}  spread {abs((vals[0] or 0) - (vals[1] or 0)):.4f}")
    for c in curves:
        print("    ", c)
for tag in ("rope-on-v", "tie-after-init"):
    d = json.loads((RUNS / f"lltGPT-{TIER}-broken-{tag}.json").read_text(encoding="utf-8"))
    print(tag, [m["valLoss"] for m in d["metrics"]], d["evals"])
