"""Timing figures for the lltSFT page. Run on an idle machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text:
  t1_steps_min    page cells for steps 1 to 7 in one namespace on the laptop CPU, minutes, whole number.
                  The T1 wall-clock on the Lab card. Models already in the Hugging Face cache; step 2 streams
                  the SmolTalk2 rows from the Hub every time, so it includes that network time.
  hand_loop_min   step 7 alone (load a fresh SmolLM2-135M, 2 held-out passes, one epoch over the packed rows), minutes, one decimal.
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

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
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
        k = step_of(path)
        if k > (3 if smoke else 7):
            break
        c0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        if k == 7:
            out["hand_loop_min"] = round((time.perf_counter() - c0) / 60, 1)
    if not smoke:
        out["t1_steps_min"] = round((time.perf_counter() - t0) / 60)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
