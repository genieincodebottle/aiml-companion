import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'e'\n" + _c["SETUP"]
RUNS = r'''
# gold reward models (Apache-2.0 Skywork V2 on Qwen3) and best-of-n sampling speed
job("gold-skywork-qwen3-1.7b-fp16", "gold_rm", "Skywork/Skywork-Reward-V2-Qwen3-1.7B", 300, seq=512)
job("gold-skywork-qwen3-4b-nf4", "gold_rm", "Skywork/Skywork-Reward-V2-Qwen3-4B", 420, seq=512, quant=True)
job("sample-smollm2-360m-instruct-n16", "sample_n", "HuggingFaceTB/SmolLM2-360M-Instruct", 300, n=16, new=256)
# TRL preference variants, 4 pairs per step at 512 tokens, Qwen3-0.6B-Base + LoRA
for v in ["ipo", "simpo", "orpo", "kto"]:
    job("trl-%s-q3-0.6b" % v, "trl_variant", "Qwen/Qwen3-0.6B-Base", 360, variant=v, steps=6)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe E: gold reward model, best-of-n sampling, TRL preference variants\n\nWrites `llt_c_probe_e.json`."), ("code", SETUP), ("code", RUNS)]
