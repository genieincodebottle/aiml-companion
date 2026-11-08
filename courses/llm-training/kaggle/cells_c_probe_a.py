import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'a'\n" + _c["SETUP"]
RUNS = r'''
# reward models (BT loss, 8 pairs/step at 512 tokens)
job("rm-smollm2-360m-full", "rm", "HuggingFaceTB/SmolLM2-360M", 300, mode="full", steps=10, pairs=8, seq=512)
job("rm-smollm2-360m-lora", "rm", "HuggingFaceTB/SmolLM2-360M", 300, mode="lora", steps=10, pairs=8, seq=512)
job("rm-q3-0.6b-lora", "rm", "Qwen/Qwen3-0.6B-Base", 300, mode="lora", steps=10, pairs=8, seq=512)
job("rm-q35-0.8b-lora", "rm", "Qwen/Qwen3.5-0.8B-Base", 360, mode="lora", steps=10, pairs=8, seq=512)
job("rm-q35-0.8b-full", "rm", "Qwen/Qwen3.5-0.8B-Base", 360, mode="full", steps=10, pairs=8, seq=512)
# DPO (manual loop, 4 pairs/step at <=512 tokens)
job("dpo-q3-0.6b-lora", "dpo", "Qwen/Qwen3-0.6B-Base", 300, mode="lora", steps=8, pairs=4, seq=512)
job("dpo-q35-0.8b-lora", "dpo", "Qwen/Qwen3.5-0.8B-Base", 360, mode="lora", steps=8, pairs=4, seq=512)
job("dpo-smollm2-360m-full", "dpo", "HuggingFaceTB/SmolLM2-360M", 300, mode="full", steps=8, pairs=4, seq=512)
job("dpo-q3-0.6b-full", "dpo", "Qwen/Qwen3-0.6B-Base", 300, mode="full", steps=8, pairs=4, seq=512)
# tool-use SFT: fp16 frozen base + fp32 LoRA, 4 x 768 per step
job("tool-q3-1.7b-lora16", "toolsft", "Qwen/Qwen3-1.7B-Base", 420, mode="lora16", steps=8, batch=4, micro=2, seq=768)
job("tool-smollm3-3b-lora16", "toolsft", "HuggingFaceTB/SmolLM3-3B-Base", 480, mode="lora16", steps=8, batch=4, micro=2, seq=768)
job("tool-q35-4b-lora16", "toolsft", "Qwen/Qwen3.5-4B-Base", 540, mode="lora16", steps=6, batch=4, micro=1, seq=768)
job("tool-q35-4b-qlora", "toolsft", "Qwen/Qwen3.5-4B-Base", 540, mode="qlora", steps=6, batch=4, micro=2, seq=768)
# TRL smoke
job("trl-rm-q3-0.6b", "trl_rm", "Qwen/Qwen3-0.6B-Base", 360, steps=6)
job("trl-dpo-q3-0.6b", "trl_dpo", "Qwen/Qwen3-0.6B-Base", 360, steps=6)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe A: reward models, DPO, tool-use SFT\n\nWrites `llt_c_probe_a.json`."), ("code", SETUP), ("code", RUNS)]
