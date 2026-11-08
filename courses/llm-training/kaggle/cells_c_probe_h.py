import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'h'\n" + _c["SETUP"]
RUNS = r'''
job("tool-q35-2b-lora16", "toolsft", "Qwen/Qwen3.5-2B-Base", 420, mode="lora16", steps=8, batch=4, micro=2, seq=768)
job("trl-grpo-q35-0.8b", "trl_grpo", "Qwen/Qwen3.5-0.8B-Base", 420, steps=3, new=128)
job("trl-dpo-q35-0.8b", "trl_dpo", "Qwen/Qwen3.5-0.8B-Base", 360, steps=6)
job("trl-rm-q35-0.8b", "trl_rm", "Qwen/Qwen3.5-0.8B-Base", 360, steps=6)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe H: Qwen3.5 under TRL, tool SFT 2B"), ("code", SETUP), ("code", RUNS)]
