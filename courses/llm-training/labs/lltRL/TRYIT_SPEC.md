# lltRL Try it spec: "Score a group, weigh its tokens"

Class A module with a Try it widget (plan section 14.2 lists "PPO clip plot" and "GRPO advantage calculator"). One widget, two linked panels, all data from `tryit_data.json` (built by `tryit_data.py` from recorded run files only; nothing runs a model in the browser).

## The one manipulation that teaches most

Pick a recorded GSM8K group (one question, 8 replies the policy actually wrote during GRPO training, each with its 0-or-1 reward and token count) and switch the **advantage rule** (GRPO with std scaling, Dr. GRPO without it, RLOO) and the **loss normalization** ("seqmean" per reply, "dapo" per batch token, "dr_grpo" constant). The learner sees:

1. The 8 advantage bars change size (not sign) between GRPO and Dr. GRPO, most for groups with 1 or 7 correct replies.
2. The per-token weight bars: under "seqmean" a long wrong reply's tokens weigh less than a short wrong reply's; under "dapo" and "dr_grpo" every token of a reply weighs the same, whatever its length. This is the length bias Liu et al. 2025 describe, on our own replies.
3. A group whose 8 rewards are equal shows all-zero bars and the label "no gradient from this group".

## Panel A: the group

- Selector: groups from `groups[]` (two per recorded step: steps 1, 50, 100, 150 of `lltRL-T2-grpo-main-s0`). Show the step, the question (first 200 characters, expandable), the gold answer.
- Table of 8 replies: reward (tick or cross), tokens (bar, with the 256 cap drawn; mark `truncated`), the reply text collapsed to one line (expandable; `text` is already cut at the stop string).
- Toggle 1, advantage rule: `advantage.grpo`, `advantage.drgrpo`, `advantage.rloo` (precomputed with the page's formulas, step 7).
- Toggle 2, normalization: `tokenWeight.seqmean`, `tokenWeight.dapo`, `tokenWeight.drgrpo`. These are computed for the GRPO-scaled advantage; the widget may recompute for the other rules as advantage / tokens / G (seqmean), advantage / total tokens in the group (dapo), advantage / (G x 256) (drgrpo).
- Readout line: "mean reward r of 8 | std s | zero spread: yes/no".

## Panel B: the run behind the groups

- Small multiples over training steps for the selected run: reward, mean reply tokens, truncated share, zero-spread share (`curves[run].reward`, `.meanLen`, `.truncated`, `.zeroStd`, per step).
- Run selector: `curves` keys (default, seed 1, seqmean, drgrpo, G4, G16, beta0.04, rloo). A vertical marker at the selected group's step.
- A second small panel, "the PPO clip": the clipped objective against the ratio (0 to 2) for advantage +1 and -1, with an epsilon slider 0.05 to 0.5 (default 0.2). Pure formula: min(r A, clip(r, 1 - e, 1 + e) A). Under it, the recorded clip fraction of our PPO run every 5 iterations (`clip.ppoClipFracByIter`, run `lltRL-T2-ppo-kl0.05-s0`), to show that about 6-10% of tokens hit the clip in practice.

## Numbers the widget prints (all from the JSON)

- Default group on load: `groups[6]` (step 100, the mother-and-daughter question, gold 26): 5 of 8 right; GRPO advantages +0.7244 and -1.2074, Dr. GRPO +0.375 and -0.625, RLOO +0.4286 and -0.7143. Per-token weight of reply 1 (191 tokens, right): seqmean 0.000474, dapo 0.000393, drgrpo 0.000354; of reply 0 (229 tokens): 0.000395, 0.000393, 0.000354. The seqmean column is the only one that changes with reply length.
- Zero-spread examples: `groups[0]`, `groups[1]` (step 1, all wrong), `groups[2]` (step 11, all right), `groups[5]`, `groups[9]` (all right).
- Curves: `curves.default.meanLen` goes from 107.6 at step 1 to 216.1 at step 50 and 179.8 at step 150; `curves.seqmean.meanLen` moves between 71 and 195 per step and averages about 120; the truncated share of the default run peaks at 0.719 at step 27. Single steps are noisy (32 replies); draw a 10-step moving mean over the raw points.
- PPO clip fraction of our run, 0.058 to 0.105 per iteration (`clip.ppoClipFracByIter`).
- The run steps 1 and 11 groups come from the start of the run and from its first step after a resume from step 10 (the A100 was preempted once); this is why one group is labelled step 11.

## Must not claim

- That a larger advantage means a better reply; it means a larger push for that group at that step.
- That "dapo" or "drgrpo" made our replies shorter: see the measured length curves; say what they show.
- Any number not in `tryit_data.json`.

## Accessibility and layout

Owner diagram style (rectangles, sentence case, no violet, green only for a correct reply). Both themes, 320 px: panels stack, the reply table scrolls vertically, never horizontally. Toggles are buttons with aria-pressed, 44 px targets. Respect reduced motion: bars change without animation.
