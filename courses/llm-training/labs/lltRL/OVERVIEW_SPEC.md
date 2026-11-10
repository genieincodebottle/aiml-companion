# lltRL overview spec (class A, delta-only)

Module 19, LLM Training Lab. The narrated overview is built later by the animation lane from this spec. Male voice, English and Spanish (D5). No TTS until the owner's go.

## What the paired overview already covers (do not repeat)

`llm-rlhf-overview` (track `llm`, module `llmRLHF`, narration in `src/components/narration/scripts/llmScripts.js` from line ~2299) teaches: base vs SFT vs RLHF on a KeyError question (30, 65, 92 helpfulness), SFT pairs and the response-only loss, comparison data, the reward model (scores -0.8 and +2.4, 70-80% agreement), ORM vs PRM, the PPO loop as "sample, score, update", the value model as a baseline (1.2) with advantages +1.2, +0.2, -1.8, reward hacking ("helpful, great, helpful") and the KL penalty with beta, DPO in one paragraph, GRPO with four samples scored 0.9, 0.2, 0.7, 0.2 and a 0.5 group baseline, the four-models-in-memory count, sycophancy, verbosity, mode collapse, Constitutional AI and RLAIF.

It never shows a clip, a ratio, a KL number, a reward-KL trade-off, a real hacked sample, a group with zero spread, or how reply length moves during RL. This overview starts there, and every number in it comes from our runs.

## The one driving question

"RL pushes a model toward a score. What stops it from pushing too far, and what does the push actually change?"

Part 1 answers the first half with PPO on IMDb (the KL penalty and the clip). Part 2 answers the second half with GRPO on GSM8K (what moved: format, length, sampled accuracy).

## Zones (6)

Every number has its run id and field. Narration rounds as in the "Spoken as" lines, nothing else. Runs live in `aiml-companion/courses/llm-training/runs/lltRL/`.

### Zone 1. One PPO step, with real objects

- Visual: an IMDb opening, "Forget what I said about Emeril," (8 tokens), the policy writing 32 tokens, the classifier returning one number. Then the four models as four boxes with their memory: policy 0.54 GB, reference 0.27, value 0.54, classifier 0.27, 1.61 GB in fp32/fp16 (`lltRL-T2-ppo-s0-job`, `evals.mem`, cumulative values 0.538, 0.807, 1.345, 1.613).
- Narration point: the score lands on the last token; every token also pays a small KL cost for moving away from the reference; GAE spreads the score back over the tokens as advantages.
- Spoken as: "four models, about one and a half gigabytes".

### Zone 2. The clip, drawn

- Visual: the PPO objective against the ratio for a positive and a negative advantage, epsilon 0.2. Mark the four cases from page cell 2: ratio 1.5 with A = +1 (flat, gradient 0), ratio 0.5 with A = +1 (gradient -0.5), ratio 1.5 with A = -1 (gradient 1.5), ratio 0.5 with A = -1 (flat, gradient 0) (`lltRL-t1-cells`, `samples[1].stdout`, lines 1-4).
- Narration point: past 1 + epsilon a good token earns no more, so one batch can be reused for several epochs without running away. A bad token that became more likely still gets pushed back.
- Do not claim the clip limits the ratio itself (see zone 3).

### Zone 3. Without the clip, and with one epoch

- Visual: three score curves over 200 iterations from the held-out checks every 25 iterations: default (clip 0.2, 4 epochs), no clip, 1 epoch (`lltRL-T2-ppo-kl0.05-s0`, `-noclip-s0`, `-epochs1-s0`, `evals.evals[].score`). Values at 100 / 200: 5.06 / 5.29, 5.02 / 5.14, 2.88 / 4.75.
- Overlay a log-scale counter of the largest ratio seen: 231 (default), 641,371 (no clip), 1.1 (one epoch) (`metrics[].ratioMax`, max over the run).
- Narration point: without the clip some updates moved a token's probability hundreds of thousands of times; the run happened to recover. With one epoch nothing moved much, and learning was slower. The clip is what makes reuse safe.
- Spoken as: "more than six hundred thousand times", "about half as far by iteration one hundred".
- One seed each: say "in our run".

### Zone 4. The KL penalty, and a hacked review

- Visual: a reward-KL plot. Points every 25 iterations for KL coefficients 0, 0.05 and 0.2 (`evals.evals[].score` against `evals.evals[].kl` in `-kl0.0-s0`, `-kl0.05-s0`, `-kl0.2-s0`). End points: (106.2, 5.67), (13.4, 5.29), (5.98, 4.25). Start (0.0009, 0.846).
- Then the text of one held-out sample at iteration 200 for coefficient 0: "...A superb, insightful and brilliantly stunning film and a great, powerful and brilliantly superb and magnificent and magnificent and exceedingly terrific and terrific..." (`-kl0.0-s0`, `evals.evals[-1].samples[2]`), next to the coefficient-0.05 sample "Forget what I said about Emeril, this one is a gem. It's an amazing movie with absolutely amazing performance..." (`-kl0.05-s0`, same index).
- Distinct-token share: 0.878 at the start, 0.597 with no penalty, 0.815 at 0.05 (`evals.evals[].distinct`).
- Spoken as: "a hundred and six nats", "thirteen", "six".

### Zone 5. GRPO: one question, eight replies, no critic

- Visual: one recorded group from the main GRPO run: the GSM8K question, 8 replies as rows with their 0-or-1 reward and token count, then the advantage bars (group mean subtracted, divided by the group's standard deviation). Source: `lltRL-T2-grpo-main-s0`, `evals.groups[]` (each entry has questions, replies, rewards, tokens, advantages). Use the step-100 group for the question "5 years ago, a mother was twice as old as her daughter. If the mother is 41 years old now, how old will the daughter be in 3 years?" (gold 26): rewards 1, 1, 0, 1, 1, 0, 0, 1; tokens 229, 191, 215, 227, 256, 242, 256, 229; GRPO advantages +0.724 for each right reply and -1.207 for each wrong one (`evals.groups[3]`, first 8 replies; also in `labs/lltRL/tryit_data.json`, `groups[6]`). Spoken as: "five of eight right; each right reply is pushed up a little, each wrong one pushed down harder".
- Then a group where all 8 replies scored the same, and its bars collapse to zero; show the share of such groups per step, `metrics[].zeroStd`, the step-150 group for the second question (gold 30): all 8 replies right, every bar 0 (`evals.groups[4]`, replies 9-16). Over the whole run 0.37 of groups had zero spread, 0.45 over the last 50 steps (`metrics[].zeroStd`). Spoken as: "by the end, almost half the groups taught nothing".
- Narration point: the group is the baseline; groups with no spread teach nothing.

### Zone 6. What the push changed

- Visual: reply length per step for the main run, seed 1, "seqmean" and "drgrpo" (`metrics[].meanLen` in `-grpo-main-s0`, `-grpo-main-s1`, `-grpo-seqmean`, `-grpo-drgrpo`), with the 256-token cap drawn and the truncated share under it (`metrics[].truncated`). Means over steps 101-150: default 184 tokens, seed 1 180, "seqmean" 120, Dr. GRPO 157 (start about 110 in every run). Truncated share for the default: 0.08 over steps 1-10, 0.31 over steps 41-50, 0.17 over steps 141-150. Spoken as: "replies nearly doubled, hit the limit, then pulled back a little". Do not attribute the difference between normalizations to the normalization alone (one seed each).
- One reply before and after training for the same test question (`lltRL-T2-grpo-main-s0`, `evals.final.replies[k]` against `lltRL-T2-grpo-base-passk`, `evals.final.replies[k]`): test question 0 (Janet's ducks, gold 18). Before: "Janet makes 16 * 2 = 32 dollars per day ... The answer is 55." After: "Let's break down Janet's earnings ... 16 - 3 - 4 = 9 eggs/day ... 9 eggs/day x $2/egg = $18/day ... Final Answer: boxed 18" (`lltRL-T2-eval-base` and `lltRL-T2-eval-main-s0`, `evals.final.replies[0]`). Show the layout change, not the LaTeX source: render the boxed 18 as a boxed number.
- End card: greedy accuracy and pass@1 / pass@8 before and after on 200 GSM8K test questions: greedy 0.495 -> 0.66 (seed 0) and 0.675 (seed 1); pass@1 0.216 -> 0.634 and 0.602; pass@8 0.620 -> 0.905 and 0.845 (`lltRL-T2-eval-base`, `-main-s0`, `-main-s1`, `evals.final`). Paired 95% interval of the greedy gain: 0.085 to 0.245 and 0.10 to 0.265 (`lltRL-T2-analysis`, `evals.pairs`). Spoken as: "about half right before, two thirds after; and sampled answers went from about one in five right to about three in five"
- Narration point (answer to the driving question): the KL penalty and the clip stop PPO from pushing too far (zones 3-4). In GRPO the push mostly made the model sure of answers it could already reach: sampled accuracy rose to near greedy, entropy fell about sevenfold, and the model adopted a step-by-step layout it already knew. Greedy accuracy still rose by about 17 points, and pass@8 rose too.

## Must not claim

- That PPO needs the clip to reach a good final score: our no-clip run ended at 5.14 against 5.29, one seed.
- That the clip bounds the ratio: the default run still had a largest ratio of 231.
- That the classifier score measures review quality: the coefficient-0 run scored highest with degenerate text.
- Any number from `llm-rlhf-overview` (30/65/92, 2.4, 1.2, 0.9/0.2/0.7/0.2).
- That GRPO taught the model to reason: we measured accuracy, layout, length and entropy, not reasoning ability. The step-by-step layout appeared within 30 steps with no SFT, so it was already in the base model. Our pass@k stops at k = 8, so the large-k question (Yue et al. 2025) is untested here.
- That our A100 times are T4 times. If a time is spoken, say "on our GPU" and use the A100 number, or skip it.
- That Qwen3 is the newest Qwen: if named, say "Qwen3-0.6B-Base".
- Internal words: "smoke", "arm", "Run Library", "lane", "wave". Say "our runs", "run", "setting".

## Spanish notes

Grammar in Spanish, every technical term in English: policy, reward model, value model, KL, clip, ratio, epoch, batch, token, advantage, reward hacking, group, pass@k, LoRA. Numbers as in the English narration rules.
