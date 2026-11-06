"""Timing figures for the lltLoRA page (CPU part). Run on an idle machine; prints ONE JSON line.

Every key fills a [[TIME:key]] placeholder in the module text:
  steps_1_to_3_s   page cells for steps 1 to 3 (LoRALinear, merge, injection into the lab GPT), seconds, whole number.
  step_4_min       page cell for step 4 (load Qwen3.5-0.8B-Base in float32 on CPU, PEFT and our layer on the
                   same 2-example batch, merge), minutes, one decimal. Assumes the checkpoint is already in the
                   Hugging Face cache (download not timed).
  cpu_lora_step_s  one LoRA training step (PEFT r16 all-linear, AdamW, float32, no autocast) on 4 GSM8K examples
                   with Qwen3.5-0.8B-Base on CPU, median of steps 2 and 3, seconds, whole number. Lab card T1 givesUp.
Run from this folder with the isolated venv: ..\\..\\.venv-latest\\Scripts\\python.exe measure_timing.py
torch.set_num_threads(4), as the cells set it. GPU timings on the page come from the Kaggle runs, not from here.
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = HERE / "steps"
sys.path.insert(0, str(STEPS))
os.chdir(STEPS)


def main():
    ns = {"__name__": "__main__"}
    out = {}
    t0 = time.perf_counter()
    for cell in ["s01_lora_linear.py", "s02_merge.py", "s03_inject_gpt.py"]:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile((STEPS / cell).read_text(encoding="utf-8"), cell, "exec"), ns)
    out["steps_1_to_3_s"] = round(time.perf_counter() - t0)
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile((STEPS / "s04_peft_check.py").read_text(encoding="utf-8"), "s04", "exec"), ns)
    out["step_4_min"] = round((time.perf_counter() - t1) / 60, 1)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    import lora_lab as L
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(L.MODEL)
    model = AutoModelForCausalLM.from_pretrained(L.MODEL, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    rows = L.encode(tok, L.load_gsm8k("train", 4))
    hist = L.train(model, rows, 3, opt, tok.pad_token_id, device="cpu", batch=4, budget=2048, warmup=1)
    out["cpu_lora_step_s"] = round(sorted(h["sec"] for h in hist[1:])[0] / 2 + sorted(h["sec"] for h in hist[1:])[1] / 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
