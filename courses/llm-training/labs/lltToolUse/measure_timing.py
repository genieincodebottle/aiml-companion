"""CPU timing of the lltToolUse page, for the quiet-machine pass. Prints one JSON line.

Runs the page's CPU cells (steps 1 to 6, cells 1 to 7, the TRL check included) in one process with 4 threads, with
the models and data already in the Hugging Face cache (a first run downloads them; run this script twice and keep the
second line). Keys:
  t1_steps_min   minutes for cells 1 to 7 (steps 1 to 6)
  t1_check_min   minutes for cell 4 alone (step 3, the TRL check with Qwen3.5-0.8B-Base on CPU)
  t1_peak_gb     peak resident memory of the process, GB
Usage: python -I measure_timing.py <dir holding an unpacked bfcl_eval package>
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
bfcl = sys.argv[1]
code = r'''
import json, os, sys, time, threading, psutil
peak = [0]
def watch():
    p = psutil.Process()
    while True:
        peak[0] = max(peak[0], p.memory_info().rss)
        time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()
sys.argv = ["run_steps.py", BFCL, "1", "2", "3", "4", "5", "6", "7"]
_timing_start = time.time()   # not t0: run_steps.py reuses t0 inside the exec'd code
__file__ = os.path.join(os.getcwd(), "run_steps.py")
exec(open("run_steps.py", encoding="utf-8").read())
total = time.time() - _timing_start
cells = json.load(open("run_steps_out.json", encoding="utf-8"))
check = [v["sec"] for k, v in cells.items() if k.startswith("cell04")][0]
print("TIMING " + json.dumps({"t1_steps_min": round(total / 60, 1), "t1_check_min": round(check / 60, 1), "t1_peak_gb": round(peak[0] / 1e9, 2)}))
'''.replace("BFCL", repr(bfcl))
p = subprocess.run([sys.executable, "-I", "-c", code], cwd=HERE, capture_output=True, text=True)
line = [l for l in p.stdout.splitlines() if l.startswith("TIMING ")]
print(line[-1][7:] if line else json.dumps({"error": (p.stderr or "")[-800:]}))
