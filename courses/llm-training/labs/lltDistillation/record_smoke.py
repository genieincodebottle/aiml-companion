"""Record the smoke kernel (llt-distill-smoke v1) as Run Library file lltDistillation-T2-smoke-stopbug. No new GPU measurement.

That run used a single stop string, newline + "Question:", and took the last number of the reply as the answer. Both base
models continue an answer with a "[Question]" block, which that stop string misses, so most replies ran to the 256-token cap
and the last number came from an invented follow-up question. This script rescoring the SAME recorded replies with the
shipped cut() and extract() (page cell 3) shows the size of the error.
"""
import json, re, sys
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import write_run  # noqa: E402

src = (HERE / "steps" / "cell03_step3.py").read_text(encoding="utf-8")
ns = {}
exec("import re" + chr(10) + src[src.index("NL = chr(10)"):src.index("tok = AutoTokenizer")] +
     src[src.index("def prompt"):src.index("def generate")], ns)
out = HERE.parent.parent / "kaggle" / "out" / "llt-distill-smoke"
R = json.load(open(out / "distill_results.json", encoding="utf-8"))
test = load_dataset("openai/gsm8k", "main")["test"].select(range(32))
rows = {}
for name, r in R.items():
    new = [ns["extract"](t) is not None and ns["extract"](t) == ns["gold"](x["answer"]) for t, x in zip(r["texts"], test)]
    rows[name] = {"accOldStopAndLastNumber": r["acc"], "accShippedCutAndExtract": round(sum(new) / len(new), 4),
                  "endedAtOldStop": r["stopped"], "endedAtShippedStops": round(sum(ns["cut"](t) != t for t in r["texts"]) / 32, 3),
                  "meanTokens": r["meanTokens"]}
    print(name, rows[name])
write_run("lltDistillation-T2-smoke-stopbug", "T2", "broken:stop-string",
          {"kernel": "llt-distill-smoke v1, 2026-11-08", "sizes": "32 test, 64 train, 2 steps of 16", "oldStop": "newline + Question:",
           "shippedStops": ["newline + Question", "newline + [Question]"], "maxNewTokens": 256},
          0, [], evals={"rescored": rows, "exampleTeacherReply": R["teacher"]["texts"][0][:700]},
          hardware={"gpu": "Tesla T4", "count": 1, "vramGB": 15.0},
          env={"python": "3.13", "torch": "2.11.0+cu130", "transformers": "5.22.1", "trl": "1.17.1", "peft": "0.21.2",
               "bitsandbytes": "0.50.2", "rescoredWith": "labs/lltDistillation/steps/cell03_step3.py cut/extract"},
          notes="Replies recorded on the T4; accuracies recomputed on the laptop from the same reply texts.")