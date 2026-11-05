"""Reproduce every exact figure on the lltGPT page (CPU, 4 threads). Timings: measure_timing.py.

Runs, in order:
  measure_steps.py     every page cell's output and the peak RAM      -> lltGPT-t1-steps
  measure_gpt2.py      GPT-2 124M logit match and the mistake table   -> lltGPT-t1-gpt2-match, lltGPT-t1-broken-conv1d-square
  measure_toggles.py   step 8, six variants x seeds 0 and 1, and the two broken training runs
                                                                       -> lltGPT-t1-toggle-*, lltGPT-t1-broken-rope-on-v, lltGPT-t1-broken-tie-after-init
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for args in (["measure_steps.py"], ["measure_gpt2.py"], ["measure_toggles.py", "--seeds", "0", "1", "--broken"]):
    print("=====", " ".join(args), flush=True)
    subprocess.run([sys.executable, str(HERE / args[0]), *args[1:]], check=True, cwd=HERE)
