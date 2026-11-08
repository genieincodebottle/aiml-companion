import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_c_probe_common.py"))
SETUP = "TAG = 'b'\n" + _c["SETUP"]
RUNS = r'''
# GRPO shape: (prompts, G). 256-token completions. fp16 base + fp32 LoRA.
job("grpo-q3-0.6b", "grpo", "Qwen/Qwen3-0.6B-Base", 480, iters=3, shapes=[[2, 4], [2, 8]], new=256, micro=4)
job("grpo-q35-0.8b", "grpo", "Qwen/Qwen3.5-0.8B-Base", 600, iters=3, shapes=[[2, 4], [2, 8]], new=256, micro=2)
job("grpo-q3-1.7b", "grpo", "Qwen/Qwen3-1.7B-Base", 600, iters=2, shapes=[[1, 8]], new=256, micro=2)
# logit KD
job("kd-q3-t4b-fp16-s0.6b-lora", "kd", "Qwen/Qwen3-0.6B-Base", 420, teacher="Qwen/Qwen3-4B-Base", teacher_quant=False, student_mode="lora", steps=6, batch=2, micro=1, seq=512)
job("kd-q3-t4b-nf4-s0.6b-full", "kd", "Qwen/Qwen3-0.6B-Base", 420, teacher="Qwen/Qwen3-4B-Base", teacher_quant=True, student_mode="full", steps=6, batch=2, micro=1, seq=512)
job("kdop-q3-t4b-nf4-s0.6b-lora", "kd_onpolicy", "Qwen/Qwen3-0.6B-Base", 480, teacher="Qwen/Qwen3-4B-Base", teacher_quant=True, student_mode="lora", steps=4, batch=8, new=128, seq=512)
job("kd-q35-t4b-fp16-s0.8b-lora", "kd", "Qwen/Qwen3.5-0.8B-Base", 540, teacher="Qwen/Qwen3.5-4B-Base", teacher_quant=False, student_mode="lora", steps=4, batch=2, micro=1, seq=512)
# TRL smoke
job("trl-grpo-q3-0.6b", "trl_grpo", "Qwen/Qwen3-0.6B-Base", 420, steps=3, new=128)
job("trl-distill-q3", "trl_distill", "Qwen/Qwen3-0.6B-Base", 480, teacher="Qwen/Qwen3-1.7B-Base", steps=3, new=128)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# LLT batch C probe B: GRPO, distillation\n\nWrites `llt_c_probe_b.json`."), ("code", SETUP), ("code", RUNS)]
