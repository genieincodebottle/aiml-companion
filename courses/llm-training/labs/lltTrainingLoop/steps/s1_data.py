import os
import urllib.request
import numpy as np
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(data_dir="data"):
    # Download once, map each character to an id, write uint8 token files.
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    if not os.path.exists(os.path.join(data_dir, "val.bin")):
        ids = np.array([stoi[c] for c in text], dtype=np.uint8)
        n = int(0.9 * len(ids))
        ids[:n].tofile(os.path.join(data_dir, "train.bin"))
        ids[n:].tofile(os.path.join(data_dir, "val.bin"))
    return chars


class Sampler:
    # Seeded batch order over a memmapped token file. Its position is two integers.
    def __init__(self, path, block, batch, seed):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.block, self.batch, self.seed = block, batch, seed
        self.n = (len(self.data) - 1 - block) // block
        self.epoch, self.pos = 0, 0
        self._shuffle()

    def _shuffle(self):
        # The order depends only on (seed, epoch), so it can be rebuilt after a restart.
        rng = np.random.default_rng([self.seed, self.epoch])
        offset = int(rng.integers(self.block))
        self.starts = offset + self.block * rng.permutation(self.n)

    def next_batch(self):
        if self.pos + self.batch > self.n:
            self.epoch, self.pos = self.epoch + 1, 0
            self._shuffle()
        s = self.starts[self.pos:self.pos + self.batch]
        self.pos += self.batch
        x = np.stack([self.data[i:i + self.block] for i in s]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.block] for i in s]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state):
        assert state["seed"] == self.seed, "sampler seed differs from the checkpoint"
        self.epoch, self.pos = state["epoch"], state["pos"]
        self._shuffle()
