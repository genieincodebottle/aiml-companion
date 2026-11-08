import os, runpy
_c = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells_rm_common.py"))
SETUP = "TAG = 'main'\n" + _c["SETUP"]
RUNS = r'''
# Part 1. Best-of-n pool: 64 held-out UltraFeedback prompts x 64 samples from SmolLM2-360M-Instruct (fp32 sampling,
# temperature 1), then the gold reward model on the pool, on 1,000 held-out pairs and on a RewardBench 2 subset.
job("pool-sample", "sample", 4200, prompts=64, n=64, promptsPerBatch=2, newTokens=256, maxRefTokens=160, seed=0)
job("pool-gold", "gold", 2400, heldPairs=1000, rb2PerSubset=60)
# Part 2. Own Bradley-Terry loss vs RewardTrainer.compute_loss on one batch, then a short RewardTrainer run.
job("trl-check", "trlcheck", 900, pairs=64, batchPairs=4, steps=40, lr=1e-5)
# Part 3. Proxy reward models: SmolLM2-360M + scalar head, full fine-tune, 3,000 UltraFeedback pairs, 1 epoch.
common = dict(pairs=3000, heldPairs=1000, batchPairs=8, scorePool=True, rb2PerSubset=60)
job("rm-s0", "train", 4200, seed=0, lr=1e-5, evalFractions=[0.25, 0.5, 1.0], evalAtZero=True, saveDir="rm-s0", **common)
job("rm-s1", "train", 3000, seed=1, lr=1e-5, evalFractions=[1.0], saveDir="rm-s1", **common)
job("rm-bal-s0", "train", 3000, seed=0, lr=1e-5, balanced=True, evalFractions=[1.0], saveDir="rm-bal-s0", **common)
# Part 4. The RewardConfig default learning rate, 1e-4, on a full fine-tune (800 pairs, 100 steps).
job("rm-lr1e-4", "train", 1200, seed=0, lr=1e-4, pairs=800, heldPairs=1000, batchPairs=8, evalFractions=[1.0])
ALL["totalMinutes"] = round((time.time() - T0) / 60, 1); save(); print("DONE", ALL["totalMinutes"])
'''
CELLS = [("md", "# lltRewardModels: best-of-n pool, gold scores and the proxy reward models\n\nOne T4 session. Writes llt_rm_main.json, pool.json, gold and proxy score files and three fp16 checkpoints."), ("code", SETUP), ("code", RUNS)]
