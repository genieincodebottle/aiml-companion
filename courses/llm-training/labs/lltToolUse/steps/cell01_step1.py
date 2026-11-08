import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # Kaggle has 2 GPUs, and Trainer refuses a model split over both
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc, json, re, time, torch
from transformers import AutoTokenizer
import tool_common as C        # written by the notebook's third cell; also labs/lltToolUse/tool_common.py in the repo

MAIN, REF = "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3-1.7B-Base"
tok, ref_tok = AutoTokenizer.from_pretrained(MAIN), AutoTokenizer.from_pretrained(REF)

weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
           "parameters": {"type": "object", "required": ["city"], "properties": {
               "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"}}}}}
ask = [{"role": "user", "content": "Weather in Pune for 3 days?"}]
call = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
    "name": "get_weather", "arguments": {"city": "Pune", "days": 3, "metric": True}}}]}

for name, t in [(MAIN, tok), (REF, ref_tok)]:
    prompt = t.apply_chat_template(ask, tools=[weather], tokenize=False, add_generation_prompt=True)
    bare = t.apply_chat_template(ask, tokenize=False, add_generation_prompt=True)
    full = t.apply_chat_template(ask + [call], tools=[weather], tokenize=False)
    extra = len(t(prompt)["input_ids"]) - len(t(bare)["input_ids"])
    print("==", name, "| the tool prompt adds", extra, "tokens | the prompt ends", repr(prompt[-30:]))
    print(full[len(prompt):])
