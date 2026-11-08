import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'g'\n" + _c["SETUP"]
RUNS = r'''
job("sample-smollm2-360m-instruct-n16-fp32", "sample_n", "HuggingFaceTB/SmolLM2-360M-Instruct", 300, n=16, new=256, fp32=True)
job("sample-smollm2-360m-instruct-n16-fp16", "sample_n", "HuggingFaceTB/SmolLM2-360M-Instruct", 300, n=16, new=256)
job("sample-q3-0.6b-n16-fp16", "sample_n", "Qwen/Qwen3-0.6B", 300, n=16, new=256)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe G: best-of-n sampling speed"), ("code", SETUP), ("code", RUNS)]
