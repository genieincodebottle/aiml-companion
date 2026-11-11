"""Time step 12 with RUN_VLLM = True on a T4, on its own. Runs steps 1 to 4 for the shared state (the model, the prompts, the
weights), takes the two helpers step 12 needs (sh from step 7, side_env from step 8) without running those steps, and expects
a merged/ folder next to it (run_extras_t4.py leaves one). Writes vllm_t4.json in the current folder.

    python run_vllm_t4.py
"""
import json, os, shutil, site, subprocess, sys, time, venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
import torch
ns = {"__name__": "__main__", "shutil": shutil, "subprocess": subprocess, "sys": sys, "site": site, "venv": venv}
steps = HERE / "steps"
read = lambda k: (steps / ("cell%02d_step%d.py" % (k, k))).read_text(encoding="utf-8")
times, t_all = {}, time.time()


def run(name, src):
    print("=====", name, time.strftime("%H:%M:%S"), flush=True)
    t = time.time()
    torch.cuda.reset_peak_memory_stats()
    exec(compile(src, name, "exec"), ns)
    times[name] = {"sec": round(time.time() - t, 1), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print("===== done", name, times[name], flush=True)


for k in (1, 2, 3, 4):
    run("cell%02d" % k, read(k).replace("FROM_OUR_CHECKPOINT = True", "FROM_OUR_CHECKPOINT = False"))
c7, c8 = read(7), read(8)
sh_src = "def sh(cmd" + c7.split("def sh(cmd", 1)[1].split("merged_sd = ", 1)[0]
side_src = "def side_env(" + c8.split("def side_env(", 1)[1].split('side_env("lc-env"', 1)[0]
exec(compile(sh_src + "\n" + side_src, "helpers", "exec"), ns)
run("cell12", read(12).replace("FROM_OUR_CHECKPOINT = True", "FROM_OUR_CHECKPOINT = False").replace("RUN_VLLM = False", "RUN_VLLM = True"))
json.dump({"cellSec": times, "totalSec": round(time.time() - t_all, 1), "vllm": ns["runs"].get("vllm")}, open("vllm_t4.json", "w"), indent=1, default=str)
