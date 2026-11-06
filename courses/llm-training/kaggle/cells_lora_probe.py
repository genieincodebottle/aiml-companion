import os, runpy
SETUP = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_lora_common.py"))["SETUP"]

PROBE = r'''
Q35 = "Qwen/Qwen3.5-0.8B-Base"; Q3 = "Qwen/Qwen3-0.6B-Base"
base = dict(steps=6, batch=8, pad_to=512, eval_final=False, n_acc=0, n_heldout=50, n_wiki=4, lr=1e-5, seed=0, warmup=2)
job(tag="q35-full8-b1024", model=Q35, mode="full", opt="adamw8bit", budget=1024, **base)
job(tag="q35-full8-b2048", model=Q35, mode="full", opt="adamw8bit", budget=2048, **base)
job(tag="q35-full32-b1024", model=Q35, mode="full", opt="adamw32", budget=1024, **base)
job(tag="q35-lora16-b2048", model=Q35, mode="lora", r=16, budget=2048, **dict(base, lr=2e-4))
job(tag="q35-lora16-b4096", model=Q35, mode="lora", r=16, budget=4096, **dict(base, lr=2e-4))
job(tag="q3-full8-b2048", model=Q3, mode="full", opt="adamw8bit", budget=2048, **base)
'''

REAL = r'''
real = dict(steps=12, batch=16, n_train=400, eval_every=6, n_heldout=300, n_wiki=32, n_acc=50, seed=0, warmup=3)
job(tag="q35-full8-real", model=Q35, mode="full", opt="adamw8bit", lr=1e-5, budget=1024, spectra=True, **real)
job(tag="q35-lora16-real", model=Q35, mode="lora", r=16, lr=2e-4, budget=2048, **real)
job(tag="q35-base-eval", model=Q35, mode="eval", n_heldout=300, n_wiki=32, n_acc=50)
print("TOTAL MIN", round((time.time() - T0) / 60, 1))
'''
CELLS = [("md", "# LLT LoRA probe\n\nMemory and speed of Qwen3.5-0.8B-Base full fine-tuning and LoRA at sequence 512 on one T4."),
         ("code", SETUP), ("code", PROBE), ("code", REAL)]
