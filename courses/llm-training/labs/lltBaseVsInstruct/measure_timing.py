"""Timing figures for the lltBaseVsInstruct page. Run on an idle machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text:
  steps_1_to_3_min   page cells for steps 1 to 3 in one namespace (load both models, 12 greedy answers of
                     40 tokens), minutes, one decimal. The quick path on the Lab card.
  notebook_min       page cells for steps 1 to 6 (adds the 40-prompt KL pass and the plain-context pass),
                     minutes, whole number. The T1 wall-clock on the Lab card.
Both assume the two SmolLM2-360M checkpoints are already in the Hugging Face cache (download not timed),
torch.set_num_threads(4) as the cells set it, float32 on CPU.
Needs `node build_notebook.mjs <module file>` first, so steps/ holds the page cells.
LLT_TIMING_SMOKE=1 runs steps 1 to 3 only, to check the script; its numbers are not for the page.
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)


def main():
    smoke = os.environ.get("LLT_TIMING_SMOKE") == "1"
    cells = sorted((HERE / "steps").glob("cell*_step*.py"))
    assert cells, "run node build_notebook.mjs first"
    step_of = lambda p: int(p.stem.split("_step")[1])
    ns = {"__name__": "__main__"}
    out = {"smoke": True} if smoke else {}
    t0 = time.perf_counter()
    for path in cells:
        if smoke and step_of(path) > 3:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        if step_of(path) == 3:
            out["steps_1_to_3_min"] = round((time.perf_counter() - t0) / 60, 1)
    if not smoke:
        out["notebook_min"] = round((time.perf_counter() - t0) / 60)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
