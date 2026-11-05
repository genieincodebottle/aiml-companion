import time
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import bpe, data

def train_bpe(vocab_size, texts, pre=None):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre or pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    return tok

text = data.train_slice(250_000)  # 250 KB of each source, 1 MB in all
start = time.perf_counter()
ours = bpe.train(text, 1024)
t_ours = time.perf_counter() - start
start = time.perf_counter()
lib = train_bpe(1024, text.splitlines())
t_lib = time.perf_counter() - start
print(f"train to 1,024 tokens: bpe.py {t_ours:.1f} s, tokenizers {t_lib:.2f} s")

held = data.heldout()
for lang in ("en", "hi"):
    cache = {}
    n_ours = sum(len(bpe.encode(x, ours, cache=cache)) for x in held[lang].splitlines())
    n_lib = sum(len(e.ids) for e in lib.encode_batch(held[lang].splitlines()))
    size = len(held[lang].encode("utf-8"))
    print(f"{lang} bytes per token: bpe.py {size / n_ours:.2f}, tokenizers {size / n_lib:.2f}")

gpt2 = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
lines = [x for lang in ("en", "hi", "es") for x in held[lang].splitlines()]
start = time.perf_counter()
cache = {}
n = sum(len(bpe.encode(x, merges, cache=cache)) for x in lines)  # GPT-2's merges, step 5
t_ours = time.perf_counter() - start
start = time.perf_counter()
n_lib = sum(len(e.ids) for e in gpt2.encode_batch(lines))
t_lib = time.perf_counter() - start
print(f"encode {n:,} tokens: bpe.py {t_ours:.2f} s, tokenizers {t_lib:.2f} s, same count {n == n_lib}")
