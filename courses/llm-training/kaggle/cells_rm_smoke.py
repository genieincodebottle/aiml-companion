import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_rm_common.py"))
SETUP = "TAG = 'smoke'\n" + _c["SETUP"]
RUNS = r'''
job("smoke-sample", "sample", 900, prompts=2, n=64, promptsPerBatch=2, newTokens=256)
job("smoke-gold", "gold", 900, heldPairs=16, rb2PerSubset=2)
job("smoke-train", "train", 900, seed=0, pairs=96, heldPairs=32, lr=1e-5, batchPairs=8, scorePool=True, evalFractions=[0.5, 1.0], evalAtZero=True, rb2PerSubset=2, saveDir="rm_smoke")
job("smoke-trl", "trlcheck", 600, pairs=16, batchPairs=4, steps=3, lr=1e-5)
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# lltRewardModels smoke: sampling, gold scoring, hand-written RM training, TRL check (small sizes)"), ("code", SETUP), ("code", RUNS)]
