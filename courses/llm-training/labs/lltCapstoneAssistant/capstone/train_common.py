"""Model loading, LoRA config, checkpoint resume and merging, shared by sft.py and dpo.py (training venv only)."""
import glob
import os

import torch

from .common import CHAT_TEMPLATE, EOS


def load_model(name, cfg, for_training=True):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = getattr(torch, cfg["dtype"])
    kw = {"dtype": dtype}
    if cfg["quant"] == "nf4" and for_training:
        # QLoRA: the frozen base is stored in 4-bit NF4, computed in fp16 on a T4 (no bf16 tensor cores)
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(name, device_map={"": 0} if torch.cuda.is_available() else None, **kw)
    if cfg["quant"] == "nf4" and for_training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def lora_config(cfg):
    from peft import LoraConfig
    return LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05, target_modules="all-linear",
                      task_type="CAUSAL_LM")


def tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.eos_token = EOS            # Qwen3-Base already uses <|endoftext|> (id 151643) as EOS and as pad
    tok.pad_token = EOS
    tok.chat_template = CHAT_TEMPLATE
    return tok


def latest_checkpoint(out_dir):
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    return cks[-1] if cks else None


def fp32_trainable(model):
    """On a T4 (fp16) every trainable LoRA weight must be fp32, or the GradScaler cannot unscale it."""
    bad = {p.dtype for p in model.parameters() if p.requires_grad} - {torch.float32}
    if bad:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    return sorted(str(d) for d in {p.dtype for p in model.parameters() if p.requires_grad})


def merge_and_save(base_name, adapter_dir, out_dir, cfg):
    """Load the base in 16-bit, add the adapter, fold it in, and save a plain model a server can load."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, GenerationConfig
    dtype = getattr(torch, cfg["dtype"])
    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    tok = tokenizer(base_name)
    model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                                               do_sample=True, temperature=0.7, top_p=0.8)
    model.config.eos_token_id = tok.eos_token_id
    model.save_pretrained(out_dir, max_shard_size="5GB")
    tok.save_pretrained(out_dir)
    return out_dir
