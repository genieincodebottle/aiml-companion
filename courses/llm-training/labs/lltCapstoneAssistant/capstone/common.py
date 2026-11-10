"""Shared settings and helpers for the lltCapstoneAssistant pipeline.

Every stage is a module you run as `python -m capstone.<stage> --tier T2|T3 --work <dir>`.
Each stage reads the previous stage's files from <work>, writes its own, and skips work that is already done,
so a stopped session (a Kaggle timeout or a Spot preemption) resumes by running the same command again.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import re
import sys
import time

# ---------------------------------------------------------------- models (checked on the Hugging Face API 2026-11-10)
BASE = "Qwen/Qwen3-1.7B-Base"            # Apache-2.0, a base model: the starting point
OFFICIAL = "Qwen/Qwen3-1.7B"             # Apache-2.0, Qwen's own post-trained model from the same base: the bar to compare against
RM = {"T2": "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "T3": "Skywork/Skywork-Reward-V2-Qwen3-8B"}   # Apache-2.0 reward models
JUDGE = {"T2": "allenai/Olmo-3-7B-Instruct", "T3": "openai/gpt-oss-20b"}                        # Apache-2.0, open judges

# The base model's chat-control tokens (<|im_start|>, <|im_end|>, <think>, </think>) share one untrained embedding row,
# and LoRA does not train embeddings, so a LoRA fine-tune cannot learn to write <|im_end|> (lltDPO, lltToolUse).
# A plain-text template that ends every reply with <|endoftext|>, a trained token, avoids the trap.
TEMPLATE = "### Instruction:\n{}\n\n### Response:\n"
EOS = "<|endoftext|>"
CHAT_TEMPLATE = (
    "{%- for m in messages %}"
    "{%- if m['role'] == 'user' %}{{ '### Instruction:\\n' + m['content'] + '\\n\\n### Response:\\n' }}"
    "{%- elif m['role'] == 'assistant' %}{{ m['content'] + '<|endoftext|>' }}"
    "{%- endif %}{%- endfor %}"
)

# ---------------------------------------------------------------- tiers: one setting per tier, the rest is shared code
TIERS = {
    # free T4 (Kaggle or Colab): 4-bit base (QLoRA), fp16 compute, Hugging Face generate, smaller data
    "T2": dict(quant="nf4", dtype="float16", backend="hf", n_sft=8000, sft_max_len=1024, sft_bs=4, sft_accum=4,
               n_pref=1500, k_samples=4, sample_max_new=512, dpo_bs=2, dpo_accum=16, dpo_max_len=1024,
               eval_max_new=768, n_arena=250, lora_r=32, lora_alpha=64),
    # one A100 40 GB or larger: bf16 base, LoRA, vLLM for every generation and for the judge
    "T3": dict(quant=None, dtype="bfloat16", backend="vllm", n_sft=20000, sft_max_len=2048, sft_bs=8, sft_accum=2,
               n_pref=5000, k_samples=4, sample_max_new=1024, dpo_bs=4, dpo_accum=8, dpo_max_len=2048,
               eval_max_new=1536, n_arena=500, lora_r=64, lora_alpha=128),
}
SMOKE = dict(n_sft=64, n_pref=24, k_samples=2, sample_max_new=256, eval_max_new=64, n_arena=8, max_steps=3)


def cli(desc, extra=None):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--tier", default="T3", choices=sorted(TIERS))
    p.add_argument("--work", default=os.path.expanduser("~/capstoneP2/work"))
    p.add_argument("--smoke", action="store_true", help="a few items and steps, to test the stage end to end")
    p.add_argument("--seed", type=int, default=0)
    for args, kw in (extra or []):
        p.add_argument(*args, **kw)
    a = p.parse_args()
    a.cfg = dict(TIERS[a.tier])
    if a.smoke:
        a.cfg.update(SMOKE)
        a.work = a.work.rstrip("/") + "-smoke"
    os.makedirs(a.work, exist_ok=True)
    seed_all(a.seed)
    return a


def seed_all(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def path(a, *parts):
    p = os.path.join(a.work, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)          # atomic: a preempted write never leaves half a file behind


def write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def done(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- text checks used by the data card
WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text, n=13):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def norm(text):
    return " ".join(WORD.findall(text.lower()))


def mostly_english(text, min_ratio=0.9):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(c.isascii() for c in letters) / len(letters) >= min_ratio


IDENTITY = re.compile(r"\b(OLMo|Olmo|Ai2|AI2|Allen Institute|T[uü]lu|ChatGPT|OpenAI|GPT-4|GPT-3\.5|Qwen|Alibaba)\b")


# ---------------------------------------------------------------- environment record for every run file
def env_versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "bitsandbytes", "vllm", "lm_eval", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            pass
    try:
        import torch
        out["cuda"] = torch.version.cuda
    except Exception:
        pass
    return out


def hardware():
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": p.name, "count": torch.cuda.device_count(), "vramGB": round(p.total_memory / 2**30, 1)}
    except Exception:
        pass
    return {"gpu": None, "count": 0, "vramGB": None, "cpu": platform.processor()}


def stage_record(a, stage, extra):
    """Append-only log of what each stage did, read later by report.py to write the Run Library files."""
    rec = {"stage": stage, "tier": a.tier, "smoke": a.smoke, "seed": a.seed, "date": datetime.datetime.utcnow().isoformat() + "Z",
           "hardware": hardware(), "env": env_versions(), "command": " ".join([os.path.basename(sys.executable)] + sys.argv)}
    rec.update(extra)
    write_json(path(a, "records", stage + ".json"), rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("env",)}, indent=1)[:3000])
    return rec


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.sec = round(time.time() - self.t0, 1)


def peak_gb():
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    except Exception:
        return None
