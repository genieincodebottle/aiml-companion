# lltRL / grpo-s0-step130

A LoRA adapter (rank 16, alpha 32, every linear layer, 10,092,544 parameters, fp32) for Qwen/Qwen3-0.6B-Base (Apache-2.0),
after 130 of 150 GRPO steps in the LLM Training Lab module lltRL, step 10. It is the default start of the notebook:
learners train the last 20 steps from here.

Training: GSM8K train (MIT), prompt "Question: <q>" + newline + "Answer:", 4 questions x 8 sampled replies per step,
256 new tokens, temperature 1, reward 1 if the final answer equals the gold answer (math-verify) else 0, group-normalized
advantages, TRL-style "dapo" token-level loss, clip 0.2, beta 0, lr 2e-5 constant, 2 minibatches per step, seed 0.
Frozen fp16 base, fp16 autocast with a GradScaler (the T4 recipe), recorded on an A100.

Run file: runs/lltRL/lltRL-T2-grpo-main-s0.json in github.com/genieincodebottle/aiml-companion.
A teaching checkpoint for grade-school maths word problems, not a general assistant.
