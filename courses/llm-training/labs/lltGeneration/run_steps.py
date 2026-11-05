"""Run the page's code blocks (steps/) in one namespace, as the notebook does, and record their output.

    python run_steps.py            # loads out/gen-model.pt (from train_small.py)
    python run_steps.py --train    # runs steps/nb_train.py instead, the notebook's own training cell

Writes runs/lltGeneration/lltGeneration-t1-notebook-steps.json with every step's printed output.
Step 7 prints timings; on a shared machine they are not quoted (measure_timing.py is the source).
"""
import argparse
import contextlib
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(name, ns):
    path = os.path.join(HERE, "steps", name)
    buf = io.StringIO()
    t = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    out = buf.getvalue()
    print("=====", name, f"({time.time() - t:.1f} s)")
    print(out, end="")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    ns = {"__name__": "__main__"}
    outputs = {}
    outputs["nb_model.py"] = run("nb_model.py", ns)
    if args.train:
        outputs["nb_train.py"] = run("nb_train.py", ns)
    else:
        from tokenizers import Tokenizer
        ck = torch.load(os.path.join(HERE, "out", "gen-model.pt"), map_location="cpu", weights_only=False)
        m = ns["GPT"](2048, 192, 4, 6, 256)
        m.load_state_dict(ck["model"])
        ns["model"] = m.eval()
        ns["tok"] = Tokenizer.from_file(os.path.join(HERE, "out", "tokenizer.json"))
    for s in ["step2.py", "step3.py", "step4.py", "step5.py", "step6.py", "step6b.py", "step7.py", "step8.py", "nb_broken.py"]:
        if s in args.skip.split(","):
            continue
        outputs[s] = run(s, ns)
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None) or mi.rss
    print("peak working set", round(peak / 1e9, 2), "GB")
    from runlog import write_run
    write_run("lltGeneration-t1-notebook-steps" + ("-train" if args.train else ""), "T1", "baseline",
              {"model": "out/gen-model.pt" if not args.train else "steps/nb_train.py", "threads": 4}, 0, [],
              evals={"stdout": outputs, "peakWorkingSetGB": round(peak / 1e9, 2),
                     "peakNote": "Windows peak working set of the whole process (psutil peak_wset), all steps in one process"},
              notes="Printed output of each page code block, run in one namespace. Step 7 timings are from a shared machine and are not quoted.")


if __name__ == "__main__":
    main()
