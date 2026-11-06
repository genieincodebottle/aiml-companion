# lltLoRA Run Library sweeps. Usage: LORA_PART=a1|a2|b|c python build_nb.py cells_lora_sweep.py <dir> <slug> "<title>"
import os, runpy, json
SETUP = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_lora_common.py"))["SETUP"]
PART = os.environ.get("LORA_PART", "a1")
CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_sweep_cfg.json")))

COMMON = r'''
Q35 = "Qwen/Qwen3.5-0.8B-Base"
POOL = ["model.layers.11.self_attn.v_proj", "model.layers.11.mlp.down_proj", "model.layers.11.self_attn.q_proj"]
C = dict(model=Q35, n_train=2000, batch=16, eval_every=25, n_heldout=300, n_wiki=32, n_acc=200, warmup=10)
FULL = dict(C, mode="full", opt="adamw8bit", budget=%(fb)d, spectra=True, pool=POOL)
LORA = dict(C, mode="lora", r=16, alpha=32, targets="all", budget=%(lb)d)
''' % {"fb": CFG["full_budget"], "lb": CFG["lora_budget"]}

JOBS = {
    "a1": r'''
job(tag="base-eval", **dict(C, mode="eval", eval_final=False, eval_base=True))
for lr in [3e-6, 1e-5, 3e-5, 1e-4]:
    job(tag=f"full-lr{lr:g}-s0", lr=lr, seed=0, **FULL)
''',
    "a2": r'''
for lr in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
    job(tag=f"lora-r16-lr{lr:g}-s0", lr=lr, seed=0, **LORA)
''',
}
for k, v in CFG.get("extra_parts", {}).items():
    JOBS[k] = v

CELLS = [("md", f"# LLT LoRA sweep, part {PART}\n\nRun Library runs for the lltLoRA lab. Writes res_*.json and svd_*.json."),
         ("code", SETUP), ("code", COMMON), ("code", JOBS[PART] + '\nprint("TOTAL MIN", round((time.time() - T0) / 60, 1))\n')]
