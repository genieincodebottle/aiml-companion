import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import bpe, data

path = hf_hub_download("gpt2", "tokenizer.json")
spec = json.load(open(path, encoding="utf-8"))
pairs = [m.split(" ") if isinstance(m, str) else m for m in spec["model"]["merges"]]
merges, vocab = bpe.merges_from_hf(pairs)
gpt2 = Tokenizer.from_file(path)
byte_of = {c: b for b, c in bpe.bytes_to_unicode().items()}

def library_tokens(text):
    return [bytes(byte_of[c] for c in t) for t in gpt2.encode(text).tokens]

texts = data.heldout_texts()
same = sum([vocab[i] for i in bpe.encode(t, merges)] == library_tokens(t) for t in texts)
print(len(merges), "merges loaded;", same, "of", len(texts), "texts give the same tokens")
