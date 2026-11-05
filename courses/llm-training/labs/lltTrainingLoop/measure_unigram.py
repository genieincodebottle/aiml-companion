"""The unigram baseline quoted in step 10: cross-entropy on the val split when every character is
predicted from its training-split frequency alone. Exact, no training.
Usage: python measure_unigram.py <data dir with train.bin and val.bin>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runlog import write_run  # noqa: E402

d = sys.argv[1]
train = np.fromfile(os.path.join(d, "train.bin"), dtype=np.uint8)
val = np.fromfile(os.path.join(d, "val.bin"), dtype=np.uint8)
counts = np.bincount(train, minlength=65).astype(np.float64)
p = counts / counts.sum()
ce = float(-np.log(p[val]).mean())
print("unigram cross-entropy on val, nats per character", round(ce, 4))
write_run("lltTrainingLoop-t1-unigram", "T1", "baseline:unigram", {"vocab": 65, "train": int(train.size), "val": int(val.size)}, None,
          [], evals={"valCrossEntropy": ce, "zeroCountCharsInVal": int((counts[val] == 0).sum())},
          notes="Character frequencies from train.bin, cross-entropy on val.bin. Used for the 3.35 figure in step 10.")
