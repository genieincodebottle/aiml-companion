"""Reference loss for step 8: predict every token by its training-set frequency, ignoring context.

Runs the page's step 8 data cell (same tokenizer and split), then scores the same 32,768 held-out
targets the step 8 val_loss uses. Writes runs/lltGPT/lltGPT-t1-unigram.json.
"""
import contextlib
import io
import math
import os
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
import torch  # noqa: E402

from runlog import write_run  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ns = {"__name__": "__main__"}
with contextlib.redirect_stdout(io.StringIO()):
    for name in ["step2", "step3", "step4", "step6", "step8"]:
        exec(compile((HERE / "steps" / f"{name}.py").read_text(encoding="utf-8"), name, "exec"), ns)
train_ids, VAL_Y = ns["train_ids"], ns["VAL_Y"]
V = ns["TOGGLE"].vocab
counts = torch.bincount(train_ids, minlength=V).double()
p = counts / counts.sum()
targets = VAL_Y.flatten()
unigram = -(p[targets].clamp_min(1e-12).log()).mean().item()
print("train tokens", len(train_ids), "| unigram cross-entropy on the 32,768 val targets", round(unigram, 4), "| ln(V)", round(math.log(V), 4))
write_run("lltGPT-t1-unigram", "T1", "reference:unigram", {"vocab": V, "valTargets": int(targets.numel())}, None, [],
          evals={"unigramValLoss": round(unigram, 4), "lnV": round(math.log(V), 4), "trainTokens": int(len(train_ids))})
