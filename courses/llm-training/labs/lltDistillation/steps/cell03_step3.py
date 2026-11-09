import gc, json, random, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

GPU = torch.device("cuda")
TEACHER, STUDENT = "Qwen/Qwen3-4B-Base", "Qwen/Qwen3-0.6B-Base"
N_TEST, N_TRAIN = 500, 2048
NL = chr(10)
STOPS = [NL + "Question", NL + "[Question]"]   # a base model answers, then starts a new question; stop there

tok = AutoTokenizer.from_pretrained(STUDENT)
tok.padding_side = "left"          # generation pads prompts on the left
same_vocab = AutoTokenizer.from_pretrained(TEACHER).get_vocab() == tok.get_vocab()
nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=nf4, dtype=torch.float16, device_map={"": 0}).eval()
print("teacher on GPU", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16).to(GPU).eval()
print("vocab size, teacher", teacher.config.vocab_size, "| student", student.config.vocab_size,
      "| tokenizer entries", len(tok), "| same token ids", same_vocab)

gsm = load_dataset("openai/gsm8k", "main")
test = [dict(q=e["question"], a=e["answer"]) for e in gsm["test"].select(range(N_TEST))]
train = [dict(q=e["question"], a=e["answer"]) for e in gsm["train"].shuffle(seed=0).select(range(N_TRAIN))]

def prompt(q):
    return "Question: " + q + NL + "Answer:"

def number(s):
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None

def cut(text):
    # the reply up to the first stop string
    ends = [text.find(s) for s in STOPS if s in text]
    return text[:min(ends)] if ends else text

def extract(text):
    # the number after the last "answer is" or "####", else the last number in the reply
    text = cut(text)
    found = re.findall(r"(?:answer is|####)[^0-9-]*(-?[0-9][0-9,]*(?:[.][0-9]+)?)", text)
    found = found or re.findall(r"-?[0-9][0-9,]*(?:[.][0-9]+)?", text)
    return number(found[-1]) if found else None

def gold(answer):
    return number(answer.split("####")[-1].strip())

def generate(model, prompts, sample=False, batch=64, max_new=256):
    # returns, per prompt, the generated token ids up to and including the stop string (pads removed)
    cfg = GenerationConfig(max_new_tokens=max_new, do_sample=sample, temperature=1.0 if sample else None, top_k=0 if sample else None,
                           top_p=1.0 if sample else None, stop_strings=STOPS, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.eos_token_id)
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(GPU)
        with torch.no_grad():
            g = model.generate(**enc, generation_config=cfg, tokenizer=tok)
        for row in g[:, enc["input_ids"].shape[1]:].tolist():
            out.append([t for t in row if t != tok.pad_token_id])
    return out

test_prompt_ids = [tok(prompt(x["q"]))["input_ids"] for x in test]
train_prompt_ids = [tok(prompt(x["q"]))["input_ids"] for x in train]
print("test", len(test), "| train", len(train), "| longest prompt", max(len(p) for p in test_prompt_ids + train_prompt_ids), "tokens")