# prep.py - a 4,096-token byte-level BPE on TinyStories, and the token files sweep.py reads.
import time
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ds = load_dataset("roneneldan/TinyStories")
train_text, val_text = ds["train"]["text"][:200_000], ds["validation"]["text"]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
t = time.time()
tok.train_from_iterator(train_text[:100_000], trainer=trainer)       # train split only
eot = tok.token_to_id("<|eot|>")

def encode(texts):
    ids = []
    for e in tok.encode_batch(texts):
        ids.extend(e.ids)
        ids.append(eot)
    return np.array(ids, dtype=np.uint16)

np.save("tok_train.npy", encode(train_text))
np.save("tok_val.npy", encode(val_text))
print(f"train {np.load('tok_train.npy').size:,} tokens, val {np.load('tok_val.npy').size:,} tokens, {time.time() - t:.0f} s")
