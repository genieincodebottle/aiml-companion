"""Run the CPU-runnable page blocks verbatim (FERT, EXT) on the local corpus copy and save their printed output.
Usage: python run_cpu_blocks.py <data_dir> <out.json>"""
import contextlib, io, json, sys, time, platform
import torch, transformers, tokenizers
torch.set_num_threads(4)
from transformers import AutoTokenizer
import dalab as L
import page_blocks as B
D, OUT = sys.argv[1:3]
rd = lambda n: [json.loads(l) for l in open(D + "/" + n + ".jsonl", encoding="utf-8")]
g = {"L": L, "AutoTokenizer": AutoTokenizer, "json": json, "torch": torch,
     "hi_test": rd("hi_test"), "hi_train": [json.loads(l) for i, l in zip(range(20000), open(D + "/hi_train.jsonl", encoding="utf-8"))],
     "tok": AutoTokenizer.from_pretrained(L.MODEL)}
R = {"date": time.strftime("%Y-%m-%d"), "env": {"python": platform.python_version(), "torch": torch.__version__,
     "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}, "blocks": {}}
for name in ("FERT", "EXT"):
    buf = io.StringIO(); t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(getattr(B, name), g)
    R["blocks"][name] = {"stdout": buf.getvalue(), "sec": round(time.time() - t, 1)}
    print(name, buf.getvalue(), flush=True)
json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
