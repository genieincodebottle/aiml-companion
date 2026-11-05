from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

repos = {"GPT-2": "gpt2", "SmolLM2": "HuggingFaceTB/SmolLM2-135M", "Qwen3": "Qwen/Qwen3-0.6B"}
toks = {name: Tokenizer.from_file(hf_hub_download(r, "tokenizer.json")) for name, r in repos.items()}
toks["ours 32k"] = ours  # from step 7

print("tokens per word    en    hi    es  code")
for name, tok in toks.items():
    row = []
    for k in ("en", "hi", "es", "code"):
        n = sum(len(e.ids) for e in tok.encode_batch(held[k].splitlines()))
        row.append(round(n / len(held[k].split()), 2))
    print(f"{name:9} {tok.get_vocab_size():7,}", row)
