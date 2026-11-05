from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, pre_tokenizers

no_regex = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
on = train_bpe(8192, data.lines(train.values()))
off = train_bpe(8192, data.lines(train.values()), pre=no_regex)
gpt2 = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))

def crossing(tok):
    learned = sorted((i, t) for t, i in tok.get_vocab().items() if i >= 256)
    return [t.replace("Ġ", " ") for i, t in learned if "Ġ" in t.lstrip("Ġ")]

for name, tok in (("gpt2", gpt2), ("regex on", on), ("regex off", off)):
    print(f"{name:9} {len(crossing(tok)):5} learned tokens cross a space, "
          f"en {bytes_per_token(tok, held['en'])} bytes per token")
print([t for t in crossing(off) if t.isascii()][:12])
