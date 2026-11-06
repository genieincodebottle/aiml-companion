import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle has 2 GPUs, and SFTTrainer refuses to split across them
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
BASE = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(BASE)
inst = AutoTokenizer.from_pretrained(BASE + "-Instruct")
print("base chat template:", tok.chat_template is not None, "| base EOS", tok.eos_token, "| base pad", tok.pad_token)
print("instruct EOS", inst.eos_token, inst.eos_token_id, "| instruct pad", inst.pad_token, inst.pad_token_id)

tok.chat_template = inst.chat_template   # borrow it: both tokenizers share all 49,152 tokens
tok.eos_token = "<|im_end|>"             # the template ends every turn with this token
tok.pad_token = "<|endoftext|>"          # keep padding apart from EOS

chat = [{"role": "user", "content": "Name a fruit."}, {"role": "assistant", "content": "A mango."}]
print(tok.apply_chat_template(chat, tokenize=False))
ids = list(tok.apply_chat_template(chat)["input_ids"])
print(len(ids), "tokens, the first", ids.index(1, 1), "of them the system prompt you did not write")
