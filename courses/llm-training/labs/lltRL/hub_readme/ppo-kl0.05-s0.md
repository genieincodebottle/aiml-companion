# lltRL / ppo-kl0.05-s0

SmolLM2-135M (HuggingFaceTB/SmolLM2-135M, Apache-2.0) after 200 iterations of hand-written PPO in the LLM Training Lab
module lltRL, step 4. Task: continue the first 8 tokens of an IMDb review with 32 tokens that read as positive. Reward:
the log-odds of the positive class of lvwerra/distilbert-imdb. KL coefficient 0.05 to the frozen starting model, clip 0.2,
4 epochs of 4 minibatches per batch of 64, lr 1e-5, GAE gamma 1 lambda 0.95, seed 0. fp32 weights, trained under fp16
autocast with a GradScaler on an A100.

On 128 held-out IMDb test openings (one sample each, fixed seed): score 0.846 -> 5.29, KL to the starting model 13.4 nats
per reply. Run file: runs/lltRL/lltRL-T2-ppo-kl0.05-s0.json in github.com/genieincodebottle/aiml-companion.

A teaching checkpoint, not a general-purpose model: it writes positive movie-review text and little else.
IMDb openings (stanfordnlp/imdb) were used only as prompts; that dataset is released for research use.
