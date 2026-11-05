from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, pre_tokenizers

split_digits = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                        pre_tokenizers.ByteLevel(add_prefix_space=False)])
single = train_bpe(32768, data.lines(train.values()), pre=split_digits)
gpt2 = Tokenizer.from_file(hf_hub_download("gpt2", "tokenizer.json"))
numbers = [" " + str(n) for n in range(10000)]

for name, tok in (("gpt2", gpt2), ("digit runs", ours), ("single digits", single)):
    counts = [len(e.ids) for e in tok.encode_batch(numbers)]
    shapes = {tuple(len(t.lstrip("Ġ")) for t in e.tokens) for e in tok.encode_batch(numbers[1000:])}
    print(f"{name:13} {sum(counts) / len(counts):.2f} tokens per number, "
          f"{len(shapes)} ways to split a 4-digit number, "
          f"code {bytes_per_token(tok, held['code'])} bytes per token")
