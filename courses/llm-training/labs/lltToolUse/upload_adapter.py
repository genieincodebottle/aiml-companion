"""Upload our seed-0 lltToolUse adapter to aimlcompanion/llm-training-lab under lltToolUse/qwen3.5-2b-lora-s0.

Usage: python -I upload_adapter.py <folder with adapter_config.json and adapter_model.safetensors> [--readme-only]
Uses the logged-in Hugging Face token (hf auth login). Uploads only the two adapter files and a short README.
"""
import os, sys, tempfile, shutil
from huggingface_hub import HfApi

src = sys.argv[1]
README = """# lltToolUse adapter, Qwen3.5-2B-Base, seed 0

Rank-16 LoRA adapter (alpha 32, every linear layer of the text model, 16,819,200 parameters) trained in the LLM
Training Lab page "Fine-Tuning for Tool Calling and Structured Output". Base model: Qwen/Qwen3.5-2B-Base, frozen in
fp16; adapters in fp32. Data: 1,200 conversations from NousResearch/hermes-function-calling-v1 (Apache-2.0), rendered
with the Qwen3.5 chat template, loss on assistant turns only. One epoch, 300 steps, learning rate 2e-4, cosine, on one
Kaggle T4 (transformers 5.22.1, trl 1.17.1, peft 0.21.2). Run file: runs/lltToolUse/lltToolUse-T2-notebook-q35-2b-s0.json
in github.com/genieincodebottle/aiml-companion.

Download the folder first, then load it from the local path. This works on Linux, macOS and Windows; passing
`subfolder=` to `PeftModel.from_pretrained` fails on Windows with PEFT 0.21.2, which joins the Hub path with a backslash.

```python
import os
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM
hub = snapshot_download("aimlcompanion/llm-training-lab", allow_patterns=["lltToolUse/qwen3.5-2b-lora-s0/*"])
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B-Base", dtype="float16")
model = PeftModel.from_pretrained(base, os.path.join(hub, "lltToolUse", "qwen3.5-2b-lora-s0"))
```
"""
tmp = tempfile.mkdtemp()
README_ONLY = "--readme-only" in sys.argv
for f in ("adapter_config.json", "adapter_model.safetensors"):
    if not README_ONLY:
        shutil.copy(os.path.join(src, f), tmp)
open(os.path.join(tmp, "README.md"), "w", encoding="utf-8").write(README)
api = HfApi()
info = api.upload_folder(folder_path=tmp, repo_id="aimlcompanion/llm-training-lab", repo_type="model",
                         path_in_repo="lltToolUse/qwen3.5-2b-lora-s0", commit_message="lltToolUse: Qwen3.5-2B-Base LoRA adapter, seed 0" + (" (README)" if README_ONLY else ""))
print(info)
