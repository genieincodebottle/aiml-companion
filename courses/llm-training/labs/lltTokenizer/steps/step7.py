from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import data

train = data.train_texts()
held = data.heldout()

def bytes_per_token(tok, text):
    n = sum(len(e.ids) for e in tok.encode_batch(text.splitlines()))
    return round(len(text.encode("utf-8")) / n, 2)

gpt2 = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
print("vocab    en    hi    es    code")
print(" gpt2", [bytes_per_token(gpt2, held[k]) for k in ("en", "hi", "es", "code")])
for size in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
    tok = train_bpe(size, data.lines(train.values()))  # train_bpe from step 6
    print(f"{size:5}", [bytes_per_token(tok, held[k]) for k in ("en", "hi", "es", "code")])
    if size == 32768:
        ours = tok
