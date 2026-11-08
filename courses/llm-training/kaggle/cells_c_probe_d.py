import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'd'\n" + _c["SETUP"]
RUNS = r'''
job("rm-q35-0.8b-lora", "rm", "Qwen/Qwen3.5-0.8B-Base", 360, mode="lora", steps=10, pairs=8, seq=512)
job("rm-q35-0.8b-full", "rm", "Qwen/Qwen3.5-0.8B-Base", 360, mode="full", steps=10, pairs=8, seq=512)
job("kd-q35-t4b-nf4-s0.8b-lora", "kd", "Qwen/Qwen3.5-0.8B-Base", 540, teacher="Qwen/Qwen3.5-4B-Base", teacher_quant=True, student_mode="lora", steps=4, batch=2, micro=1, seq=512)
job("grpo-q3-0.6b-bigbatch", "grpo", "Qwen/Qwen3-0.6B-Base", 600, iters=2, shapes=[[4, 8], [8, 8]], new=256, micro=8)
job("trl-grpo-tools-q3-0.6b", "trl_grpo_tools", "Qwen/Qwen3-0.6B", 420, steps=3, new=256)
job("trl-distill-q3", "trl_distill", "Qwen/Qwen3-0.6B-Base", 480, teacher="Qwen/Qwen3-1.7B-Base", steps=3, new=128)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe D: fixes + GRPO batch scaling + tool-call GRPO\n\nWrites `llt_c_probe_d.json`."), ("code", SETUP), ("code", RUNS)]
